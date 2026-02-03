from __future__ import annotations

import io, os, time, asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

import httpx
import torch
from transformers import CLIPProcessor, CLIPModel

# ================= ENV =================
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# ================= CONFIG =================
MODEL_NAME = "openai/clip-vit-base-patch32"

SEARCH_SOURCES = os.getenv("SEARCH_SOURCES", "naver,openverse").split(",")

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

OPENVERSE_CLIENT_ID = os.getenv("OPENVERSE_CLIENT_ID")
OPENVERSE_CLIENT_SECRET = os.getenv("OPENVERSE_CLIENT_SECRET")

PORTRAIT_AR_MIN = 1.02
BBOX_FEET_Y_MIN = 0.86

ITEM_TOP1_MIN_PROB = 0.18
ITEM_MARGIN_MIN = 0.03

COLOR_STRICT = False

MAX_CONCURRENCY = 10
FINAL_LIMIT_DEFAULT = 8

# ================= APP =================
app = FastAPI(title="Styling Recommend API")

# ================= MODELS =================
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# YOLO
from ultralytics import YOLO
yolo_person = YOLO("yolov8n.pt")

# ================= UTILS =================
def pil_rgb(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def clamp(v, a, b): 
    return max(a, min(b, v))

# ================= COLOR =================
def color_compatible(user_color: str, cand_color: str) -> bool:
    u = (user_color or "").lower().strip()
    c = (cand_color or "").lower().strip()
    if not u or not c:
        return False
    if u == c:
        return True

    neutral = {"black", "white", "gray", "beige", "brown"}
    warm = {"red", "orange", "yellow", "pink", "brown", "beige"}
    cool = {"blue", "green", "purple"}

    if u == "white":
        return c in {"white", "gray", "beige", "black"}
    if u == "black":
        return c in {"black", "gray", "white"}
    if u == "gray":
        return c in {"gray", "black", "white"}
    if u == "beige":
        return c in {"beige", "white", "brown", "gray"}

    if u in warm:
        return c in warm or c in neutral
    if u in cool:
        return c in cool or c in neutral

    return False

# ================= BODY CHECK =================
def bbox_fullbody_and_feet(img: Image.Image) -> Tuple[bool, Dict]:
    w, h = img.size
    r = yolo_person.predict(img, verbose=False)[0]
    if r.boxes is None:
        return False, {}

    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    persons = [i for i, c in enumerate(cls) if int(c) == 0]
    if not persons:
        return False, {}

    i = max(persons, key=lambda i: (boxes[i][3]-boxes[i][1])*(boxes[i][2]-boxes[i][0]))
    x1, y1, x2, y2 = boxes[i]

    person_h = (y2 - y1) / h
    top_ratio = y1 / h
    bottom_ratio = y2 / h

    full_ok = person_h >= 0.7
    feet_ok = bottom_ratio >= BBOX_FEET_Y_MIN
    head_ok = top_ratio <= 0.15

    return full_ok and feet_ok and head_ok, {
        "person_h": float(person_h),
        "top_ratio": float(top_ratio),
        "bottom_ratio": float(bottom_ratio),
        "full_ok": int(full_ok),
        "feet_ok": int(feet_ok),
        "head_ok": int(head_ok),
    }

# ================= CLIP =================
@torch.no_grad()
def clip_scores(img: Image.Image, prompts: List[str]) -> List[float]:
    inp = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    out = clip_model(**inp)
    probs = out.logits_per_image.softmax(dim=1)[0]
    return probs.cpu().tolist()

# ================= NAVER IMAGE =================
async def naver_search(query: str, display=50):
    url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json().get("items", [])

# ================= MAIN =================
@app.post("/recommend/image")
async def recommend_image(
    image: UploadFile = File(...),
    limit: int = Form(FINAL_LIMIT_DEFAULT),
    requestId: str = Form(...)
):
    user_img = pil_rgb(await image.read())

    category_prompts = [
        "a photo of footwear",
        "a photo of top clothing",
        "a photo of bottom clothing",
        "a photo of outerwear",
        "a photo of an accessory",
    ]
    cat_scores = clip_scores(user_img, category_prompts)
    category = category_prompts[int(torch.tensor(cat_scores).argmax())]

    subtype_prompts = [
        "a photo of leather dress shoes",
        "a photo of dress shoes",
        "a photo of derby shoes",
        "a photo of monk strap shoes",
        "a photo of oxford shoes",
    ]
    subtype_scores = clip_scores(user_img, subtype_prompts)
    top_idx = int(torch.tensor(subtype_scores).argmax())
    target_prompt = subtype_prompts[top_idx]
    baseLabel = target_prompt.replace("a photo of ", "")

    korean_query = "남자 구두 코디 전신 스트릿룩 데일리룩"
    naver_items = await naver_search(korean_query)

    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process(it):
        async with sem:
            try:
                img = pil_rgb((await httpx.AsyncClient().get(it["link"])).content)
            except:
                return None

        ok, bbox = bbox_fullbody_and_feet(img)
        if not ok:
            return None

        ar = img.height / img.width
        ar_score = ar if ar >= PORTRAIT_AR_MIN else 0.9

        match_scores = clip_scores(img, subtype_prompts)
        top1 = max(match_scores)
        margin = top1 - sorted(match_scores)[-2]

        penalty = 0.0
        if top1 < ITEM_TOP1_MIN_PROB:
            penalty += 0.12
        if margin < ITEM_MARGIN_MIN:
            penalty += 0.1

        score = 0.6 * top1 + 0.25 * ar_score + 0.15 * bbox["person_h"]
        score -= penalty

        return {
            "imageUrl": it["link"],
            "imageUrlDirect": it["link"],
            "landingUrl": it["link"],
            "thumbnailUrl": it["thumbnail"],
            "title": it.get("title"),
            "source": "naver",
            "score": clamp(score, 0, 2),
            "debug": {
                "matchTop1Prob": top1,
                "matchMargin": margin,
                "arScore": ar_score,
                "bodyScore": bbox["person_h"],
                "bbox": bbox,
            }
        }

    processed = await asyncio.gather(*[process(it) for it in naver_items])
    items = [x for x in processed if x]
    items.sort(key=lambda x: x["score"], reverse=True)

    return {
        "requestId": requestId,
        "query": korean_query,
        "category": category.replace("a photo of ", ""),
        "baseLabel": baseLabel,
        "targetPrompt": target_prompt,
        "rawCount": len(naver_items),
        "itemsCount": len(items[:limit]),
        "items": items[:limit],
    }
