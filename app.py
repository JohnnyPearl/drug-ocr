"""MediFind OCR service.

FastAPI microservice that reads a Vietnamese prescription image, groups the
recognised text into one block per prescribed drug, and matches each drug
against a reference drug database (with optional LLM-assisted correction).
"""

import glob
import json
import logging
import os
import re
import tempfile
import time
import unicodedata
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
from paddleocr import PaddleOCR
from starlette.concurrency import run_in_threadpool

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("medifind_ocr")

app = FastAPI(
    title="MediFind OCR API",
    description="OCR for Vietnamese prescriptions (Azure Computer Vision Read / PaddleOCR)",
)


# ======= Configuration =======
def clean_env(name: str, default: str = "") -> str:
    """Read an env var, stripping inline comments and whitespace."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.split("#", 1)[0].strip()


AZURE_OCR_ENDPOINT = clean_env("AZURE_OCR_ENDPOINT").rstrip("/")
AZURE_OCR_KEY = clean_env("AZURE_OCR_KEY")
AZURE_OCR_REGION = clean_env("AZURE_OCR_REGION")
AZURE_OCR_POLL_INTERVAL = float(clean_env("AZURE_OCR_POLL_INTERVAL", "1"))
AZURE_OCR_MAX_POLLS = int(clean_env("AZURE_OCR_MAX_POLLS", "15"))
OCR_PROVIDER = clean_env("OCR_PROVIDER").lower()
SKIP_PADDLE_OCR_INIT = clean_env("SKIP_PADDLE_OCR_INIT") == "1"

OPENAI_API_KEY = clean_env("OPENAI_API_KEY")
OPENAI_MODEL = clean_env("OPENAI_MODEL", "gpt-4-turbo")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if openai_client is None:
    logger.warning("OPENAI_API_KEY not set; LLM correction is disabled")

# PaddleOCR is initialised lazily on first use (and can be skipped in tests).
OCR_INIT_KWARGS = dict(
    lang="vi",
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)
ocr = None

# Set to True to apply CLAHE + median blur before PaddleOCR (helps dark/blurry photos).
USE_PREPROCESS = False

DRUGS_DB_PATH = Path(__file__).parent / "data" / "drugs_data.json"

# Dosage units as they appear on Vietnamese prescriptions (with and without accents).
UNITS = [
    "vien", "viên",
    "goi", "gói",
    "ong", "ống",
    "chai",
    "hop", "hộp",
    "vi", "vỉ",
    "lo", "lọ",
    "tuyp", "tuýp",
]
UNIT_CANONICAL = {
    "vien": "viên",
    "goi": "gói",
    "ong": "ống",
    "chai": "chai",
    "hop": "hộp",
    "vi": "vỉ",
    "lo": "lọ",
    "tuyp": "tuýp",
}

# Below this fuzzy-match score, the LLM is asked to pick among the top candidates.
LLM_THRESHOLD = 0.7


# ======= Text helpers =======
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_text(s: str) -> str:
    """Lowercase, strip accents and collapse whitespace for matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9+\-/\s]", " ", strip_accents(s).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9\-+]+", text) if t]


UNIT_PATTERN = "|".join(sorted(set(normalize_text(u) for u in UNITS)))
qty_pat = re.compile(
    r"(?P<num>\d+(?:[\/]\d+)?)\s*(?P<unit>(" + UNIT_PATTERN + r"))\b",
    flags=re.IGNORECASE,
)
num_head_pat = re.compile(r"^\s*\d+\s*[\.\)]?$")


# ======= Drug database =======
def load_drugs_database() -> List[Dict[str, Any]]:
    """Load drug names from drugs_data.json and precompute matching keys."""
    entries = []
    try:
        if DRUGS_DB_PATH.exists():
            drugs_data = json.loads(DRUGS_DB_PATH.read_text(encoding="utf-8"))
            for drug in drugs_data:
                if not isinstance(drug, dict):
                    continue
                # "tenThuoc" (drug name) is the field used by the source dataset.
                name = drug.get("tenThuoc") or drug.get("name")
                if not name:
                    continue
                norm = normalize_text(name)
                entries.append({
                    "name": name,
                    "normalized": norm,
                    "tokens": set(tokenize(norm)),
                    "raw": drug,
                })
    except Exception as e:
        logger.error("Error loading drugs database: %s", e)
    return entries


DRUGS_DATABASE = load_drugs_database()
logger.info("Loaded %d drugs from database", len(DRUGS_DATABASE))


# ======= Structured extraction =======
def extract_quantity(lines: List[str]) -> Optional[Dict[str, str]]:
    """Find the first quantity/unit pair in the provided lines."""
    for line in lines:
        match = qty_pat.search(normalize_text(line))
        if match:
            unit_raw = match.group("unit")
            return {"value": match.group("num"), "unit": UNIT_CANONICAL.get(unit_raw, unit_raw)}
    return None


def extract_usage(lines: List[str]) -> str:
    """Concatenate lines that look like usage instructions ("Ngày uống 3 lần...")."""
    usage_lines = []
    for line in lines:
        norm = normalize_text(line)
        if any(token in norm for token in ["ngay", "uong", "lan"]):
            usage_lines.append(line.strip())
    return " ".join(usage_lines).strip()


def clean_name_text(line: str) -> str:
    """Remove numbering/quantity from the line to isolate the drug name."""
    seg = line.split("|")[0]
    seg = re.sub(r"^\s*\d+\s*[\.\)]?\s*", "", seg).strip()
    # Remove trailing quantity fragments (best-effort)
    if qty_pat.search(normalize_text(seg)):
        seg = re.sub(r"\s+\d+.*$", "", seg).strip()
    return seg.strip(" |-")


def compute_match_score(raw_norm: str, raw_tokens: set, entry: Dict[str, Any]) -> float:
    """Hybrid score using sequence similarity and token overlap."""
    ratio = SequenceMatcher(None, raw_norm, entry["normalized"]).ratio()
    union = raw_tokens | entry["tokens"]
    inter = raw_tokens & entry["tokens"]
    token_score = len(inter) / len(union) if union else 0.0
    prefix_match = (
        entry["normalized"].startswith(raw_norm[:6])
        or raw_norm.startswith(entry["normalized"][:6])
    )
    prefix_bonus = 0.1 if prefix_match else 0.0
    return min(1.0, 0.65 * ratio + 0.35 * token_score + prefix_bonus)


def find_top_candidates(raw_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
    norm = normalize_text(raw_name)
    tokens = set(tokenize(norm))
    scored = []
    for entry in DRUGS_DATABASE:
        score = compute_match_score(norm, tokens, entry)
        if score > 0:
            scored.append((entry["name"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def choose_with_llm(raw_name: str, candidates: List[str]) -> Optional[str]:
    """Ask the LLM to pick the best candidate name for an OCR string."""
    if not openai_client or not candidates:
        return None
    try:
        prompt = (
            "You are a pharmacist. Choose the most likely correct drug name from the "
            "candidate list based on the OCR text. Return JSON only: "
            "{\"chosen_name\": \"<a name from the list, or the OCR text if none fits>\"}.\n"
            f"OCR: {raw_name}\n"
            f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
        )
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You pick the best matching drug name from the candidate list. If none fits, return the OCR text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("chosen_name")
    except Exception as e:
        logger.warning("LLM rerank error: %s", e)
        return None


def resolve_drug_name(raw_name: str) -> Tuple[str, float, str]:
    """Return best matching drug name, score, and source ('fuzzy', 'llm' or 'none')."""
    candidates = find_top_candidates(raw_name, top_k=5)
    if not candidates:
        return raw_name, 0.0, "none"
    best_name, best_score = candidates[0]
    source = "fuzzy"
    if best_score < LLM_THRESHOLD:
        llm_choice = choose_with_llm(raw_name, [c[0] for c in candidates])
        if llm_choice:
            # Keep the fuzzy score to signal the confidence floor.
            best_name = llm_choice
            source = "llm"
    return best_name, best_score, source


def build_structured_drugs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    structured = []
    for block in blocks:
        lines = block.get("lines", [])
        if not lines:
            continue
        raw_name = clean_name_text(lines[0])
        matched_name, score, source = resolve_drug_name(raw_name)
        structured.append({
            "drugName": matched_name,
            "rawName": raw_name,
            "quantity": extract_quantity(lines) or {},
            "usageText": extract_usage(lines),
            "matchScore": round(score, 3),
            "source": source,
            "blockIndex": block.get("index"),
        })
    return structured


def verify_and_correct_drugs_with_chatgpt(ocr_blocks: List[Dict]) -> Dict[str, Any]:
    """Ask the LLM to correct drug names in the OCR blocks against the drug database."""
    if not openai_client:
        logger.info("LLM not configured, returning OCR result unchanged")
        return {"verified": False, "blocks": ocr_blocks}

    try:
        # Keep the prompt compact to stay within the context limit.
        top_names = [d["name"] for d in DRUGS_DATABASE[:200]]
        drugs_list_str = json.dumps(top_names, ensure_ascii=False)

        ocr_text = json.dumps(ocr_blocks, ensure_ascii=False)
        if len(ocr_text) > 4000:
            ocr_text = ocr_text[:4000] + "...(truncated)"

        prompt = f"""You are a pharmaceutical expert. For each OCR block, choose the correct drug name from the candidate list.

CANDIDATE LIST:
{drugs_list_str}

OCR RESULT (may contain errors):
{ocr_text}

Return JSON:
{{
  "corrections": [
    {{
      "original_text": "original OCR text",
      "corrected_name": "drug name chosen from the candidates",
      "confidence": 0.9
    }}
  ],
  "corrected_blocks": [...]
}}

Return JSON only, with no other text. If no candidate fits, keep the OCR name."""

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You pick drug names from the provided candidate list and return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )

        response_text = response.choices[0].message.content
        logger.debug("LLM response: %s", response_text)

        try:
            result = json.loads(response_text)
            return {
                "verified": True,
                "corrections": result.get("corrections", []),
                "blocks": result.get("corrected_blocks", ocr_blocks),
                "raw_response": response_text,
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON")
            return {"verified": False, "blocks": ocr_blocks, "response": response_text}

    except Exception as e:
        logger.exception("Error calling LLM")
        return {"verified": False, "blocks": ocr_blocks, "error": str(e)}


# ======= OCR box handling =======
def enhance_image(image_array: np.ndarray) -> np.ndarray:
    """Light preprocessing: CLAHE for local contrast, then median blur for salt-and-pepper noise."""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.medianBlur(clahe.apply(gray), 3)


def points_to_box(points: List[float]) -> Dict[str, float]:
    """Convert a flat [x, y, x, y, ...] list into an axis-aligned box."""
    xs, ys = points[0::2], points[1::2]
    l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    w, h = max(1, r - l), max(1, b - t)
    return {"l": l, "t": t, "r": r, "b": b, "w": w, "h": h, "xc": l + w / 2, "yc": t + h / 2}


def polygon_to_box(bbox: List[float]) -> Optional[Dict[str, float]]:
    if not bbox or len(bbox) != 8:
        return None
    xs, ys = bbox[0::2], bbox[1::2]
    l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    w, h = max(1.0, r - l), max(1.0, b - t)
    return {"l": l, "t": t, "r": r, "b": b, "w": w, "h": h, "xc": l + w / 2, "yc": t + h / 2}


def load_boxes_from_paddleocr(result) -> List[Dict[str, Any]]:
    """Convert PaddleOCR predict() results into text boxes via their JSON export."""
    if not result:
        return []

    boxes = []
    with tempfile.TemporaryDirectory() as temp_dir:
        json_paths = []
        for i, res in enumerate(result):
            try:
                res.save_to_json(temp_dir)
                json_files = sorted(glob.glob(f"{temp_dir}/*.json"))
                json_paths.append(json_files[-1] if json_files else None)
            except Exception as e:
                logger.warning("Error saving JSON for result %d: %s", i, e)

        for json_path in json_paths:
            if not json_path or not Path(json_path).exists():
                continue
            try:
                json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                if isinstance(json_data, dict) and "rec_texts" in json_data and "rec_boxes" in json_data:
                    for txt, box in zip(json_data["rec_texts"], json_data["rec_boxes"]):
                        if not txt or not txt.strip():
                            continue
                        boxes.append({"text": txt.strip(), **points_to_box(box)})
            except Exception as e:
                logger.warning("Error reading JSON %s: %s", json_path, e)

    logger.info("Loaded %d boxes from PaddleOCR result", len(boxes))
    return boxes


def azure_result_to_boxes(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    if not result:
        return boxes

    analyze_result = result.get("analyzeResult") or {}
    for page in analyze_result.get("readResults", []):
        for line in page.get("lines", []):
            words = line.get("words") or []
            # Prefer word-level boxes to keep downstream grouping consistent
            if words:
                for word in words:
                    text = (word.get("text") or "").strip()
                    box = polygon_to_box(word.get("boundingBox") or line.get("boundingBox"))
                    if text and box:
                        boxes.append({"text": text, **box})
            else:
                text = (line.get("text") or "").strip()
                box = polygon_to_box(line.get("boundingBox"))
                if text and box:
                    boxes.append({"text": text, **box})

    logger.info("Loaded %d boxes from Azure OCR result", len(boxes))
    return boxes


# ======= Layout grouping =======
def group_rows(boxes, y_tol_factor=0.6):
    """Group boxes into text rows by vertical centre."""
    boxes = sorted(boxes, key=lambda b: b["yc"])
    heights = [b["h"] for b in boxes]
    med_h = sorted(heights)[len(heights) // 2] if heights else 20
    y_tol = max(8, y_tol_factor * med_h)

    rows, current, last_y = [], [], None
    for b in boxes:
        if last_y is None or abs(b["yc"] - last_y) <= y_tol:
            current.append(b)
            last_y = b["yc"] if last_y is None else (last_y * 0.7 + b["yc"] * 0.3)
        else:
            rows.append(sorted(current, key=lambda x: x["l"]))
            current = [b]
            last_y = b["yc"]
    if current:
        rows.append(sorted(current, key=lambda x: x["l"]))
    return rows, med_h


def merge_number_unit(row):
    """Merge adjacent number + unit tokens within a row (e.g. "20" + "viên")."""
    merged, i = [], 0
    while i < len(row):
        a = row[i]
        txt = strip_accents(a["text"].lower())
        if i + 1 < len(row):
            b = row[i + 1]
            txt_b = strip_accents(b["text"].lower())
            if txt.isdigit() and txt_b in UNITS:
                merged.append({**a, "text": f"{a['text']} {b['text']}"})
                i += 2
                continue
            if txt_b.isdigit() and txt in UNITS:
                merged.append({**a, "text": f"{b['text']} {a['text']}"})
                i += 2
                continue
        merged.append(a)
        i += 1
    return merged


def row_bounds(row):
    """Return (top, bottom) of a row based on its boxes."""
    return min(b["t"] for b in row), max(b["b"] for b in row)


def row_starts_item(row):
    """Check whether a row starts with an item number ("1.", "2)", ...)."""
    if not row:
        return False
    left_sorted = sorted(row, key=lambda b: b["l"])
    s1 = left_sorted[0]["text"].strip()
    s2 = left_sorted[1]["text"].strip() if len(left_sorted) > 1 else ""
    if num_head_pat.match(s1):
        return True
    if s1.isdigit() and s2 in {".", ")"}:
        return True
    return bool(re.match(r"^\s*\d+\s*[\.\)]", s1))


def build_item_blocks(rows, med_h, gap_factor=2.0, min_gap_px=25):
    """Split rows into blocks, one per prescribed drug item."""
    blocks = []
    cur = []
    for r in rows:
        if not cur:
            cur = [r]
            continue

        # Start a new block on an item number or a large vertical gap.
        cut_by_number = row_starts_item(r)
        _, cur_bottom = row_bounds(cur[-1])
        next_top, _ = row_bounds(r)
        gap_tol = max(gap_factor * med_h, min_gap_px)
        cut_by_gap = (next_top - cur_bottom) > gap_tol

        if cut_by_number or cut_by_gap:
            blocks.append(cur)
            cur = [r]
        else:
            cur.append(r)

    if cur:
        blocks.append(cur)
    return blocks


def build_result_from_boxes(boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    filtered = [b for b in boxes if len(str(b.get("text", "")).strip()) >= 1]
    if not filtered:
        return {"id": str(uuid.uuid4())[:12], "blocks": [], "structured_drugs": []}

    rows, med_h = group_rows(filtered)
    rows = [merge_number_unit(r) for r in rows]

    blocks_raw = build_item_blocks(rows, med_h, gap_factor=2.0, min_gap_px=25)
    result_blocks = []
    for i, block in enumerate(blocks_raw, 1):
        lines = []
        for row in block:
            sorted_tokens = sorted(row, key=lambda x: x["l"])
            line_text = " | ".join(t["text"] for t in sorted_tokens if str(t["text"]).strip())
            if line_text.strip():
                lines.append(line_text)
        if lines:
            result_blocks.append({"index": i, "lines": lines})

    return {
        "id": str(uuid.uuid4())[:12],
        "blocks": result_blocks,
        "structured_drugs": build_structured_drugs(result_blocks),
    }


# ======= OCR providers =======
class AzureOcrError(Exception):
    pass


def get_ocr_provider() -> str:
    if OCR_PROVIDER in {"azure", "paddle"}:
        return OCR_PROVIDER
    if AZURE_OCR_ENDPOINT and AZURE_OCR_KEY:
        return "azure"
    return "paddle"


def run_paddle_ocr(image_array: np.ndarray) -> List[Dict[str, Any]]:
    global ocr
    if image_array is None:
        raise ValueError("Empty or invalid image for PaddleOCR")

    if ocr is None:
        if SKIP_PADDLE_OCR_INIT:
            raise RuntimeError("PaddleOCR initialization skipped via SKIP_PADDLE_OCR_INIT")
        ocr = PaddleOCR(**OCR_INIT_KWARGS)

    input_for_ocr = enhance_image(image_array) if USE_PREPROCESS else image_array
    return load_boxes_from_paddleocr(ocr.predict(input_for_ocr))


def call_azure_ocr(image_bytes: bytes) -> Dict[str, Any]:
    if not AZURE_OCR_ENDPOINT or not AZURE_OCR_KEY:
        raise AzureOcrError("Azure OCR not configured (AZURE_OCR_ENDPOINT/AZURE_OCR_KEY)")
    if not image_bytes:
        raise AzureOcrError("No image bytes provided for Azure OCR")

    analyze_url = f"{AZURE_OCR_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_OCR_KEY,
        "Content-Type": "application/octet-stream",
    }
    if AZURE_OCR_REGION:
        headers["Ocp-Apim-Subscription-Region"] = AZURE_OCR_REGION

    try:
        response = requests.post(analyze_url, headers=headers, data=image_bytes, timeout=30)
    except Exception as exc:
        raise AzureOcrError(f"Azure analyze request failed: {exc}") from exc

    if response.status_code not in (200, 202):
        raise AzureOcrError(f"Azure analyze failed: {response.status_code} {response.text}")

    operation_url = response.headers.get("Operation-Location")
    if not operation_url:
        raise AzureOcrError("Azure analyze response missing Operation-Location header")

    poll_headers = {"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY}
    poll_interval = max(0.1, AZURE_OCR_POLL_INTERVAL)
    max_polls = max(1, AZURE_OCR_MAX_POLLS)

    for _ in range(max_polls):
        time.sleep(poll_interval)
        try:
            poll_resp = requests.get(operation_url, headers=poll_headers, timeout=30)
        except Exception as exc:
            raise AzureOcrError(f"Azure poll failed: {exc}") from exc

        if poll_resp.status_code >= 400:
            raise AzureOcrError(f"Azure poll error: {poll_resp.status_code} {poll_resp.text}")

        result = poll_resp.json()
        status = (result.get("status") or "").lower()
        if status == "succeeded":
            return result
        if status == "failed":
            raise AzureOcrError("Azure OCR reported failure")

    raise AzureOcrError("Azure OCR polling timed out")


def run_azure_ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    return azure_result_to_boxes(call_azure_ocr(image_bytes))


def process_ocr_result(image_array: np.ndarray, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Run OCR with the configured provider and return the normalised result."""
    provider = get_ocr_provider()
    logger.info("OCR provider selected: %s", provider)

    if provider == "azure":
        if not image_bytes and image_array is not None:
            ok, buffer = cv2.imencode(".jpg", image_array)
            if not ok:
                raise AzureOcrError("Failed to encode image for Azure OCR")
            image_bytes = buffer.tobytes()
        boxes = run_azure_ocr(image_bytes)
    elif provider == "paddle":
        boxes = run_paddle_ocr(image_array)
    else:
        raise RuntimeError(f"Unsupported OCR provider: {provider}")

    return build_result_from_boxes(boxes)


# ======= API =======
async def read_image_upload(file: UploadFile) -> Tuple[np.ndarray, bytes]:
    """Validate an uploaded image and decode it with OpenCV."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image, contents


async def run_ocr_safely(file: UploadFile) -> Dict[str, Any]:
    """Read the upload, run OCR, and map provider/unexpected errors to HTTP errors."""
    try:
        image, contents = await read_image_upload(file)
        result = await run_in_threadpool(process_ocr_result, image, contents)
        logger.info("OCR result: %d blocks found", len(result.get("blocks", [])))
        return result
    except HTTPException:
        raise
    except AzureOcrError as e:
        logger.error("OCR provider error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    """Run OCR on a prescription image and return text blocks and structured drugs."""
    return JSONResponse(content=await run_ocr_safely(file))


@app.post("/ocr-with-verification")
async def process_ocr_with_verification(file: UploadFile = File(...)):
    """Run OCR, then ask the LLM to verify/correct drug names against the database."""
    ocr_result = await run_ocr_safely(file)
    verified_result = await run_in_threadpool(
        verify_and_correct_drugs_with_chatgpt, ocr_result.get("blocks", [])
    )
    return JSONResponse(content={
        "id": ocr_result.get("id"),
        "ocr_blocks": ocr_result.get("blocks", []),
        "structured_drugs": ocr_result.get("structured_drugs", []),
        "verification": verified_result,
    })


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "MediFind OCR API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is working properly"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
