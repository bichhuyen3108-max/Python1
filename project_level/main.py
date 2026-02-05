from __future__ import annotations

import io, os, asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image
import traceback

import httpx
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# ================= ENV =================
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# ================= CONFIG =================
MODEL_NAME = "openai/clip-vit-base-patch32"

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# ✅ 전신 필터
PORTRAIT_AR_MIN = 1.02
BBOX_FEET_Y_MIN = 0.86  # (legacy) 아래 hard/soft에서 사용하지 않지만 남겨둠

# ✅ 아이템 존재 gate (region crop에서 target vs distractor)
ITEM_REGION_MIN_PROB = 0.20
ITEM_REGION_MARGIN_MIN = 0.02

# ✅ 색상 (정확 우선)
COLOR_STRICT_DEFAULT = True
COLOR_FALLBACK_COMPAT = False

# ✅ POSE gate: 너무 빡세면 0개가 되므로 "soft" 기본
POSE_GATE_STRICT = False      # True면 pose fail은 무조건 drop
POSE_GATE_PENALTY = 0.10      # soft일 때 pose fail이면 score에서 패널티

# ✅ screenshot/product gate margin: 너무 빡세면 0개가 되므로 완만하게
OUTFIT_BAD_MARGIN = 0.00      # outfit_score > bad_score + margin

# ✅ HEAD-TO-TOE(신발/발영역) gate  ⭐️(추가)
REQUIRE_FOOTWEAR_VISIBLE = True   # True: 신발/발영역이 의미있게 안 보이면 drop
FOOTWEAR_MIN_PROB = 0.18          # feet crop에서 shoe/leg 확률 최소
FOOTWEAR_MARGIN_MIN = 0.01        # shoe/leg vs bad(배경/바닥) margin
FOOTWEAR_PENALTY = 0.10           # (soft 모드일 때) 신발 애매하면 점수 패널티
FOOTWEAR_BONUS = 0.08             # 신발 잘 보이면 점수 보너스

# ✅ bbox feet hard/soft  ⭐️(추가)
BBOX_FEET_Y_HARD_MIN = 0.88       # 이 아래면 무조건 drop (발이 너무 잘림)
BBOX_FEET_Y_SOFT = 0.92           # 0.88~0.92면 살리되 점수 패널티
FEET_SOFT_PENALTY = 0.08

MAX_CONCURRENCY = 6
FINAL_LIMIT_DEFAULT = 8

# ================= APP =================
app = FastAPI(title="Styling Recommend API")

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    print("\n=== UNHANDLED ERROR ===")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": repr(exc)})

# ================= MODELS =================
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

from ultralytics import YOLO
yolo_person = YOLO("yolov8n.pt")
yolo_pose = YOLO("yolov8n-pose.pt")

# rembg
try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
except Exception:
    REMBG_OK = False


# ================= UTILS =================
def pil_rgb(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def clamp(v, a, b):
    return max(a, min(b, v))

def looks_like_image_contenttype(ct: str) -> bool:
    ct = (ct or "").lower()
    return ct.startswith("image/")

def safe_crop(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    x1 = int(clamp(int(x1), 0, w - 1))
    y1 = int(clamp(int(y1), 0, h - 1))
    x2 = int(clamp(int(x2), x1 + 1, w))
    y2 = int(clamp(int(y2), y1 + 1, h))
    return img.crop((x1, y1, x2, y2))

def center_square_crop(img: Image.Image, ratio: float = 0.85) -> Image.Image:
    w, h = img.size
    side = int(min(w, h) * ratio)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    return safe_crop(img, x1, y1, x2, y2)

def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("a photo of ", "").strip()
    if s.startswith("a "): s = s[2:].strip()
    if s.startswith("an "): s = s[3:].strip()
    return s


# ================= CLIP =================
@torch.no_grad()
def clip_scores(img: Image.Image, prompts: List[str]) -> List[float]:
    inp = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    out = clip_model(**inp)
    probs = out.logits_per_image.softmax(dim=1)[0]
    return probs.cpu().tolist()

@torch.no_grad()
def get_img_features(img: Image.Image) -> torch.Tensor:
    inp = clip_processor(images=img, return_tensors="pt")
    pixel_values = inp["pixel_values"].to(device)

    feats = None
    try:
        feats = clip_model.get_image_features(pixel_values=pixel_values)
    except Exception:
        feats = None

    if feats is None or not torch.is_tensor(feats):
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        feats = clip_model.visual_projection(pooled)

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# ================= COLOR (rembg + LAB) =================
COLOR_LABELS = ["black", "white", "gray", "beige", "brown", "navy"]
COLOR_KOR = {
    "beige": "베이지",
    "black": "검정",
    "white": "흰색",
    "navy": "네이비",
    "gray": "회색",
    "brown": "브라운",
}

def rgb_to_lab_image(img_rgb: Image.Image) -> np.ndarray:
    lab = img_rgb.convert("LAB")
    return np.asarray(lab, dtype=np.float32)

def lab_prototypes() -> Dict[str, np.ndarray]:
    # ✅ beige/white 경계용 프로토타입
    proto_rgb = {
        "black": (15, 15, 15),
        "white": (245, 245, 245),
        "gray":  (160, 160, 160),
        "beige": (210, 190, 155),
        "brown": (120, 85, 55),
        "navy":  (25, 40, 85),
    }
    out = {}
    for k, rgb in proto_rgb.items():
        p = Image.new("RGB", (1, 1), rgb)
        out[k] = rgb_to_lab_image(p).reshape(3)
    return out

LAB_PROTOS = lab_prototypes()

def rembg_alpha(img_rgb: Image.Image) -> np.ndarray:
    if not REMBG_OK:
        raise RuntimeError("rembg 미설치/로드 실패 (pip install rembg onnxruntime)")
    buf = io.BytesIO()
    img_rgb.save(buf, format="PNG")
    out_bytes = rembg_remove(buf.getvalue())
    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    alpha = np.asarray(out_img.split()[-1], dtype=np.float32) / 255.0
    return alpha

def dominant_color_label_rembg_lab(img_rgb: Image.Image) -> Tuple[str, float]:
    """
    ✅ (한국어) rembg로 배경 제거 후 LAB 중앙값 → 프로토타입 거리로 색 결정
    """
    a = rembg_alpha(img_rgb)
    lab = rgb_to_lab_image(img_rgb)

    fg = a > 0.35
    if fg.sum() < 80:
        img2 = center_square_crop(img_rgb, 0.80)
        lab2 = rgb_to_lab_image(img2)
        med = np.median(lab2.reshape(-1, 3), axis=0)
    else:
        med = np.median(lab[fg].reshape(-1, 3), axis=0)

    dists = {k: float(np.linalg.norm(med - p)) for k, p in LAB_PROTOS.items()}
    best = min(dists, key=dists.get)
    conf = float(clamp(1.0 - dists[best] / 60.0, 0.0, 1.0))
    return best, conf

def color_compatible(user_color: str, cand_color: str) -> bool:
    u = (user_color or "").lower().strip()
    c = (cand_color or "").lower().strip()
    if not u or not c:
        return False
    if u == c:
        return True
    neutral = {"black", "white", "gray", "beige", "brown", "navy"}
    if u in {"white", "black", "gray"}:
        return c in neutral
    if u == "beige":
        return c in {"beige", "brown", "white", "gray"}
    if u == "brown":
        return c in {"brown", "beige", "black", "navy"}
    if u == "navy":
        return c in {"navy", "black", "gray", "white"}
    return False


# ================= YOLO PERSON + POSE FULLBODY =================
def yolo_best_person_bbox(img: Image.Image) -> Tuple[Optional[Tuple[float,float,float,float]], int]:
    r = yolo_person.predict(img, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    persons = [i for i, c in enumerate(cls) if int(c) == 0]
    if not persons:
        return None, 0
    i = max(persons, key=lambda k: (boxes[k][2]-boxes[k][0])*(boxes[k][3]-boxes[k][1]))
    x1, y1, x2, y2 = boxes[i]
    return (float(x1), float(y1), float(x2), float(y2)), len(persons)

def bbox_fullbody_and_feet(img: Image.Image) -> Tuple[bool, Dict]:
    """
    ✅ (한국어) bbox 기반 전신 + 발(하단) 체크
    - hard: bottom_ratio < BBOX_FEET_Y_HARD_MIN 이면 drop
    - soft: BBOX_FEET_Y_HARD_MIN ~ BBOX_FEET_Y_SOFT 구간은 살리되 score penalty
    """
    w, h = img.size
    bbox, person_count = yolo_best_person_bbox(img)
    if bbox is None:
        return False, {"person_count": 0}
    x1, y1, x2, y2 = bbox
    person_h = (y2 - y1) / h
    top_ratio = y1 / h
    bottom_ratio = y2 / h

    full_ok = person_h >= 0.70
    feet_ok = bottom_ratio >= BBOX_FEET_Y_HARD_MIN  # ⭐️ hard 기준
    head_ok = top_ratio <= 0.15
    multi_ok = (person_count == 1)

    ok = (full_ok and feet_ok and head_ok and multi_ok)

    return ok, {
        "person_h": float(person_h),
        "top_ratio": float(top_ratio),
        "bottom_ratio": float(bottom_ratio),
        "full_ok": int(full_ok),
        "feet_ok": int(feet_ok),
        "head_ok": int(head_ok),
        "multi_ok": int(multi_ok),
        "person_count": int(person_count),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
    }

def pose_fullbody_gate(img: Image.Image) -> Tuple[bool, Dict[str, Any]]:
    """
    ✅ (한국어) pose로 "머리+발"이 보이는지 체크
    - pose 모델은 실패하는 경우가 있어서 기본은 soft gate로 사용
    """
    w, h = img.size
    r = yolo_pose.predict(img, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return False, {"pose": "no_keypoints"}

    xy = r.keypoints.xy.cpu().numpy()     # [n,k,2]
    conf = r.keypoints.conf.cpu().numpy() # [n,k]
    n = xy.shape[0]

    def area_of(i: int) -> float:
        pts = xy[i]; c = conf[i]
        m = c > 0.2
        if m.sum() < 4:
            return 0.0
        p = pts[m]
        return float((p[:,0].max()-p[:,0].min())*(p[:,1].max()-p[:,1].min()))

    best_i = max(range(n), key=area_of)
    pts = xy[best_i]
    cfs = conf[best_i]

    # COCO: nose=0, left_ankle=15, right_ankle=16
    if cfs[0] <= 0.20:
        return False, {"pose": "nose_missing"}
    la_ok = cfs[15] > 0.20
    ra_ok = cfs[16] > 0.20
    if not (la_ok or ra_ok):
        return False, {"pose": "ankle_missing"}

    nose_y = float(pts[0,1] / max(1, h))
    if la_ok and ra_ok:
        ankle_y = float(max(pts[15,1], pts[16,1]) / max(1, h))
    else:
        ankle_y = float((pts[15,1] if la_ok else pts[16,1]) / max(1, h))

    head_ok = nose_y <= 0.22
    feet_ok = ankle_y >= 0.90
    ok = head_ok and feet_ok

    return ok, {"nose_y": round(nose_y,4), "ankle_y": round(ankle_y,4), "head_ok": int(head_ok), "feet_ok": int(feet_ok)}

def crop_region_by_part(person_crop: Image.Image, part: str) -> Image.Image:
    """
    ✅ (한국어) item이 존재할 가능성이 높은 부위를 crop
    part: head / upper / lower / feet / hands / torso
    """
    w, h = person_crop.size
    part = part or "torso"

    if part == "head":
        return safe_crop(person_crop, 0, h*0.00, w, h*0.25)
    if part == "upper":
        return safe_crop(person_crop, 0, h*0.12, w, h*0.55)
    if part == "lower":
        return safe_crop(person_crop, 0, h*0.45, w, h*0.95)
    if part == "feet":
        return safe_crop(person_crop, 0, h*0.70, w, h*1.00)
    if part == "hands":
        return safe_crop(person_crop, 0, h*0.25, w, h*0.85)
    return safe_crop(person_crop, 0, h*0.12, w, h*0.85)


# ================= TAXONOMY (ANY ITEM 지원 핵심) =================
ITEM_TAXONOMY: List[Dict[str, Any]] = [
    # shoes
    {"en": "sneakers", "kr": ["운동화", "스니커즈"], "part": "feet", "yolo_classes": []},
    {"en": "boots", "kr": ["부츠"], "part": "feet", "yolo_classes": []},
    {"en": "loafers", "kr": ["로퍼"], "part": "feet", "yolo_classes": []},
    {"en": "dress shoes", "kr": ["구두", "드레스슈즈"], "part": "feet", "yolo_classes": []},

    # tops
    {"en": "t-shirt", "kr": ["티셔츠", "반팔"], "part": "upper", "yolo_classes": []},
    {"en": "shirt", "kr": ["셔츠"], "part": "upper", "yolo_classes": []},
    {"en": "hoodie", "kr": ["후드", "후드티"], "part": "upper", "yolo_classes": []},
    {"en": "knit sweater", "kr": ["니트", "스웨터"], "part": "upper", "yolo_classes": []},

    # bottoms
    {"en": "jeans", "kr": ["청바지", "데님"], "part": "lower", "yolo_classes": []},
    {"en": "slacks", "kr": ["슬랙스"], "part": "lower", "yolo_classes": []},
    {"en": "shorts", "kr": ["반바지"], "part": "lower", "yolo_classes": []},
    {"en": "skirt", "kr": ["치마", "스커트"], "part": "lower", "yolo_classes": []},

    # outerwear
    {"en": "long coat", "kr": ["롱코트", "코트"], "part": "torso", "yolo_classes": []},
    {"en": "trench coat", "kr": ["트렌치코트", "트렌치"], "part": "torso", "yolo_classes": []},
    {"en": "puffer jacket", "kr": ["패딩", "푸퍼"], "part": "torso", "yolo_classes": []},
    {"en": "blazer", "kr": ["블레이저", "자켓"], "part": "upper", "yolo_classes": []},

    # bags
    {"en": "backpack", "kr": ["백팩", "배낭"], "part": "hands", "yolo_classes": ["backpack"]},
    {"en": "handbag", "kr": ["핸드백", "숄더백", "토트백"], "part": "hands", "yolo_classes": ["handbag"]},
    {"en": "crossbody bag", "kr": ["크로스백"], "part": "hands", "yolo_classes": []},

    # accessories
    {"en": "hat", "kr": ["모자", "캡"], "part": "head", "yolo_classes": []},
    {"en": "sunglasses", "kr": ["선글라스"], "part": "head", "yolo_classes": []},
    {"en": "tie", "kr": ["넥타이"], "part": "upper", "yolo_classes": ["tie"]},
    {"en": "umbrella", "kr": ["우산"], "part": "hands", "yolo_classes": ["umbrella"]},
    {"en": "watch", "kr": ["시계"], "part": "hands", "yolo_classes": []},
]

def taxonomy_prompts() -> List[str]:
    return [f"a photo of {x['en']}" for x in ITEM_TAXONOMY]

def taxonomy_best(user_crop: Image.Image) -> Tuple[Dict[str, Any], float]:
    prompts = taxonomy_prompts()
    scores = clip_scores(user_crop, prompts)
    idx = int(torch.tensor(scores).argmax())
    best_item = ITEM_TAXONOMY[idx]
    best_prob = float(scores[idx])
    return best_item, best_prob


# ================= NAVER IMAGE =================
async def naver_search_once(query: str, display: int = 80, start: int = 1) -> List[Dict[str, Any]]:
    """
    ✅ (한국어) Naver image search 1회
    - sort=sim: 유사도 우선
    - filter=large: 가능하면 큰 이미지 선호 (품질/전신 확률↑)
    """
    url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": "sim",
        "filter": "large",
    }

    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(url, headers=headers, params=params)
        print("[NAVER]", r.status_code, "| query =", query)
        if r.status_code != 200:
            print("[NAVER ERR]", r.text[:300])
            return []
        data = r.json()
        return data.get("items", []) or []

def merge_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        link = it.get("link")
        if not link or link in seen:
            continue
        out.append(it)
        seen.add(link)
    return out

async def naver_search_multi(queries: List[str], display_each: int = 80) -> List[Dict[str, Any]]:
    """
    ✅ (한국어) query 여러 개를 호출해서 합치기
    - rawCount=0 문제를 줄이는 핵심
    """
    all_items: List[Dict[str, Any]] = []
    for q in queries:
        items1 = await naver_search_once(q, display=display_each, start=1)
        all_items.extend(items1)
    return merge_dedupe(all_items)


# ================= OPTIONAL YOLO OBJECT GATE =================
YOLO_NAMES = yolo_person.model.names if hasattr(yolo_person, "model") and hasattr(yolo_person.model, "names") else {}

def yolo_has_any(img: Image.Image, wanted_names: List[str]) -> bool:
    if not wanted_names:
        return True
    r = yolo_person.predict(img, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return False
    cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
    present = set(YOLO_NAMES.get(i, "") for i in cls)
    return any(w in present for w in wanted_names)


# ================= MAIN =================
@app.post("/recommend/image")
async def recommend_image(
    image: UploadFile = File(...),
    limit: int = Form(FINAL_LIMIT_DEFAULT),
    requestId: str = Form(...),
    textQuery: str = Form("")
):
    print("\n[REQ] HIT /recommend/image")
    print("[REQ] requestId=", requestId, "textQuery=", textQuery, "limit=", limit)
    print("[REQ] filename=", image.filename, "content_type=", image.content_type)

    if not NAVER_CLIENT_ID:
        return {"error": "NAVER_CLIENT_ID 없음"}
    if not NAVER_CLIENT_SECRET:
        return {"error": "NAVER_CLIENT_SECRET 없음"}
    if not REMBG_OK:
        return {"error": "rembg 미설치/로드 실패 (pip install rembg onnxruntime 필요)"}

    user_img = pil_rgb(await image.read())
    user_q = (textQuery or "").strip()

    # ✅ drop counters
    drop_counts = defaultdict(int)
    drop_lock = asyncio.Lock()
    async def drop(reason: str):
        async with drop_lock:
            drop_counts[reason] += 1

    # =========================================================
    # 0) user_mode 판단
    # =========================================================
    user_person_bbox, user_person_count = yolo_best_person_bbox(user_img)
    if user_person_bbox is not None and user_person_count >= 1:
        user_person_crop = safe_crop(user_img, *user_person_bbox)
        user_base_for_clip = user_person_crop
        user_mode = "person_wearing"
    else:
        user_person_crop = None
        user_base_for_clip = center_square_crop(user_img, 0.85)
        user_mode = "item_only"

    # =========================================================
    # 1) ANY-ITEM 분류
    # =========================================================
    best_item, best_prob = taxonomy_best(user_base_for_clip)

    item_en = best_item["en"]
    item_kr_list = best_item["kr"]
    item_part = best_item["part"]
    yolo_gate_classes = best_item.get("yolo_classes", [])

    # =========================================================
    # 2) 사용자 아이템 crop(색/특징)
    # =========================================================
    if user_person_crop is not None:
        user_item_crop = crop_region_by_part(user_person_crop, item_part)
    else:
        user_item_crop = center_square_crop(user_img, 0.85)

    user_color, user_color_conf = dominant_color_label_rembg_lab(user_item_crop)
    user_item_features = get_img_features(user_item_crop)

    # =========================================================
    # 3) 검색 쿼리 생성 (rawCount=0 방지용 다중 쿼리)
    # =========================================================
    color_kor = COLOR_KOR.get(user_color, "")
    item_kor = item_kr_list[0] if item_kr_list else "패션"

    q1 = f"{color_kor} {item_kor} 전신 착샷 코디 룩북 무신사 스냅 {user_q}".strip()
    q2 = f"{color_kor} {item_kor} 전신 코디 착샷 룩북 {user_q}".strip()
    q3 = f"{item_kor} 전신 착샷 코디 룩북 무신사 스냅 {user_q}".strip()
    q4 = f"{item_kor} 전신 코디 착샷 룩북 {user_q}".strip()
    queries = [q1, q2, q3, q4]

    naver_items = await naver_search_multi(queries, display_each=80)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # =========================================================
    # 4) 후보 처리
    # =========================================================
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:

        async def process(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with sem:
                try:
                    url = it.get("link")
                    if not url:
                        await drop("no_url")
                        return None

                    r = await client.get(url)
                    if r.status_code != 200:
                        await drop("download_fail")
                        return None

                    ct = r.headers.get("content-type", "")
                    if not looks_like_image_contenttype(ct):
                        await drop("not_image_contenttype")
                        return None

                    try:
                        img = pil_rgb(r.content)
                    except Exception:
                        await drop("decode_fail")
                        return None

                    # bbox fullbody (hard feet)
                    ok_bbox, bbox = bbox_fullbody_and_feet(img)
                    if not ok_bbox:
                        await drop("bbox_fullbody_fail")
                        return None

                    bottom_ratio = float(bbox.get("bottom_ratio", 0.0))
                    feet_soft_bad = (bottom_ratio < BBOX_FEET_Y_SOFT)  # soft zone penalty

                    # pose fullbody (soft by default)
                    ok_pose, pose_meta = pose_fullbody_gate(img)
                    if POSE_GATE_STRICT and (not ok_pose):
                        await drop("pose_fullbody_fail")
                        return None

                    # screenshot/product gate
                    gate_prompts = [
                        "a full body street fashion outfit photo",
                        "a product photo of an item",
                        "a screenshot of a webpage",
                        "a news photo with text overlay",
                    ]
                    gs = clip_scores(img, gate_prompts)
                    outfit_score = float(gs[0])
                    bad_score = float(max(gs[1], gs[2], gs[3]))
                    if not (outfit_score > bad_score + OUTFIT_BAD_MARGIN):
                        await drop("screenshot_or_product")
                        return None

                    pb = bbox.get("bbox")
                    if not pb:
                        await drop("bbox_missing")
                        return None
                    person_crop = safe_crop(img, pb[0], pb[1], pb[2], pb[3])

                    # =========================================================
                    # ✅ Head-to-toe 핵심: feet region에서 신발/발영역이 보이는지 체크
                    # =========================================================
                    feet_crop = crop_region_by_part(person_crop, "feet")
                    foot_prompts = [
                        "a photo of shoes",
                        "a photo of sneakers",
                        "a photo of boots",
                        "a photo of loafers",
                        "a photo of dress shoes",
                        "a photo of pants",
                        "a photo of legs",
                        "a photo of floor",
                        "a photo of background",
                    ]
                    fs = clip_scores(feet_crop, foot_prompts)

                    shoe_score = float(max(fs[0], fs[1], fs[2], fs[3], fs[4]))
                    leg_score = float(max(fs[5], fs[6]))
                    bad_feet = float(max(fs[7], fs[8]))

                    feet_top1 = float(max(shoe_score, leg_score))
                    feet_margin = float(feet_top1 - bad_feet)
                    footwear_visible_ok = (feet_top1 >= FOOTWEAR_MIN_PROB) and (feet_margin >= FOOTWEAR_MARGIN_MIN)

                    if REQUIRE_FOOTWEAR_VISIBLE and (not footwear_visible_ok):
                        await drop("footwear_not_visible")
                        return None

                    # OPTIONAL YOLO object gate
                    if yolo_gate_classes:
                        if not yolo_has_any(person_crop, yolo_gate_classes):
                            await drop("yolo_object_missing")
                            return None

                    # item region crop
                    cand_item_crop = crop_region_by_part(person_crop, item_part)

                    # item presence gate: target vs distractors
                    distractors = []
                    if "bag" in item_en or item_en in ("backpack", "handbag", "crossbody bag"):
                        distractors = ["handbag", "backpack", "tote bag", "crossbody bag", "suitcase"]
                    elif "shoe" in item_en or item_en in ("sneakers", "boots", "loafers", "dress shoes"):
                        distractors = ["sneakers", "boots", "loafers", "dress shoes", "sandals"]
                    elif item_part == "upper":
                        distractors = ["t-shirt", "shirt", "hoodie", "knit sweater", "jacket"]
                    elif item_part == "lower":
                        distractors = ["jeans", "slacks", "shorts", "skirt"]
                    else:
                        distractors = ["clothing", "fashion item", "outfit"]

                    candidates_en = [item_en] + [d for d in distractors if d != item_en][:4]
                    prompts = [f"a photo of {x}" for x in candidates_en]
                    rs = clip_scores(cand_item_crop, prompts)

                    region_top1 = float(max(rs))
                    srt = sorted(rs, reverse=True)
                    region_margin = float(srt[0] - srt[1]) if len(srt) >= 2 else 0.0

                    if region_top1 < ITEM_REGION_MIN_PROB:
                        await drop("item_region_low")
                        return None
                    if region_margin < ITEM_REGION_MARGIN_MIN:
                        await drop("item_region_ambiguous")
                        return None

                    # color strict
                    cand_color, cand_color_conf = dominant_color_label_rembg_lab(cand_item_crop)
                    strict_color_ok = (cand_color == user_color) if user_color else False
                    compat_color_ok = color_compatible(user_color, cand_color) if user_color else False

                    if COLOR_STRICT_DEFAULT and not strict_color_ok:
                        await drop("color_strict_mismatch")
                        return None

                    # visual similarity
                    cand_item_features = get_img_features(cand_item_crop)
                    visual_sim = float((user_item_features @ cand_item_features.T).item())

                    # portrait score
                    ar = img.height / max(1, img.width)
                    ar_score = ar if ar >= PORTRAIT_AR_MIN else 0.90

                    # style score
                    style_score = 0.0
                    if user_q:
                        style_prompts = [
                            f"a full body {user_q} outfit photo",
                            f"a {user_q} street fashion lookbook",
                            f"a fashion style of {user_q}",
                        ]
                        style_scores = clip_scores(person_crop, style_prompts)
                        style_score = float(max(style_scores))

                    score = (
                        0.45 * visual_sim +
                        0.25 * region_top1 +
                        0.10 * style_score +
                        0.10 * ar_score +
                        0.10 * float(bbox.get("person_h", 0.0))
                    )

                    # ✅ pose soft penalty
                    if not ok_pose:
                        score -= POSE_GATE_PENALTY

                    # ✅ feet soft penalty (bbox bottom이 살짝 잘린 경우)
                    if feet_soft_bad:
                        score -= FEET_SOFT_PENALTY

                    # ✅ footwear bonus/penalty (soft 모드에서도 품질↑)
                    if footwear_visible_ok:
                        score += FOOTWEAR_BONUS
                    else:
                        score -= FOOTWEAR_PENALTY

                    title = (it.get("title") or "").replace("<b>", "").replace("</b>", "")

                    # ✅ (핵심) landingUrl / thumbUrl 추가
                    landing = it.get("originallink") or url
                    thumb = it.get("thumbnail") or url

                    return {
                        "imageUrl": url,
                        "landingUrl": landing,
                        "thumbUrl": thumb,
                        "title": title,
                        "source": "naver",
                        "score": clamp(score, 0, 2),
                        "meta": {
                            "user_mode": user_mode,
                            "detected_item_en": item_en,
                            "detected_item_kr": item_kor,
                            "detected_item_prob": round(best_prob, 4),
                            "item_part": item_part,
                            "yolo_gate_classes": yolo_gate_classes,
                            "user_color": user_color,
                            "user_color_conf": round(float(user_color_conf), 4),
                            "cand_color": cand_color,
                            "cand_color_conf": round(float(cand_color_conf), 4),
                            "strict_color_ok": int(strict_color_ok),
                            "compat_color_ok": int(compat_color_ok),
                            "region_top1": round(region_top1, 4),
                            "region_margin": round(region_margin, 4),
                            "visual_sim": round(visual_sim, 4),
                            "bbox": bbox,
                            "pose": pose_meta,
                            "gate_scores": {"outfit": round(outfit_score,4), "bad": round(bad_score,4)},
                            "feet_gate": {
                                "bottom_ratio": round(bottom_ratio, 4),
                                "feet_soft_bad": int(feet_soft_bad),
                                "shoe_score": round(shoe_score, 4),
                                "leg_score": round(leg_score, 4),
                                "bad_score": round(bad_feet, 4),
                                "feet_top1": round(feet_top1, 4),
                                "feet_margin": round(feet_margin, 4),
                                "footwear_visible_ok": int(footwear_visible_ok),
                                "require_footwear_visible": int(REQUIRE_FOOTWEAR_VISIBLE),
                            },
                        }
                    }

                except Exception:
                    await drop("process_exception")
                    return None

        processed = await asyncio.gather(*[process(it) for it in naver_items])

    items = [x for x in processed if x]
    items.sort(key=lambda x: x["score"], reverse=True)
    final = items[:limit]

    for idx, it in enumerate(final, start=1):
        it["rank"] = idx

    return {
        "requestId": requestId,
        "items": final,
        "debug": {
            "detected": {
                "user_mode": user_mode,
                "item_en": item_en,
                "item_kr": item_kor,
                "item_prob": round(best_prob, 4),
                "user_color": user_color,
                "user_color_conf": round(float(user_color_conf), 4),
                "queries": queries,
                "yolo_gate_classes": yolo_gate_classes,
                "part": item_part,
                "pose_gate_strict": POSE_GATE_STRICT,
                "outfit_bad_margin": OUTFIT_BAD_MARGIN,
                "require_footwear_visible": REQUIRE_FOOTWEAR_VISIBLE,
                "footwear_min_prob": FOOTWEAR_MIN_PROB,
                "footwear_margin_min": FOOTWEAR_MARGIN_MIN,
                "bbox_feet_hard_min": BBOX_FEET_Y_HARD_MIN,
                "bbox_feet_soft": BBOX_FEET_Y_SOFT,
            },
            "rawCount": len(naver_items),
            "passCount": len(items),
            "finalCount": len(final),
            "drop_counts": dict(sorted(drop_counts.items(), key=lambda kv: kv[1], reverse=True)),
        }
    }

# 실행:
# python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 --log-level debug
