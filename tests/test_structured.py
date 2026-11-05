import os
import pytest

os.environ.setdefault("SKIP_PADDLE_OCR_INIT", "1")

from app import build_structured_drugs


SAMPLE_BLOCKS = [
    {
        "index": 1,
        "lines": [
            "1. YESOM 40 40mg (Esomeprazol 40mg) | 63 Viên",
            "Ngày uống 1 lần, mỗi lần 1 viên",
        ],
    },
    {
        "index": 2,
        "lines": [
            "2. SUCRATE GEL (Sucralfate 1g/gói) | 63 gói",
            "Ngày uống 3 lần, mỗi lần 1 gói",
        ],
    },
]


def test_build_structured_drugs_extracts_quantity_and_usage():
    structured = build_structured_drugs(SAMPLE_BLOCKS)
    assert len(structured) == 2

    first = structured[0]
    assert first["quantity"]["value"] == "63"
    assert first["quantity"]["unit"].lower().startswith("vi")
    assert "ngày uống 1 lần" in first["usageText"].lower()


def test_build_structured_drugs_matches_known_brand():
    structured = build_structured_drugs(SAMPLE_BLOCKS)
    yesom = structured[0]["drugName"].lower()
    # Accept either the branded or generic mapping, but must include yesom/esomeprazole
    assert "yesom" in yesom or "esomeprazole" in yesom
