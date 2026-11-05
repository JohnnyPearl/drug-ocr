import os

os.environ.setdefault("SKIP_PADDLE_OCR_INIT", "1")

from app import azure_result_to_boxes, build_result_from_boxes


AZURE_SAMPLE = {
    "status": "succeeded",
    "analyzeResult": {
        "readResults": [
            {
                "page": 1,
                "angle": 0,
                "width": 600,
                "height": 800,
                "unit": "pixel",
                "lines": [
                    {
                        "text": "Paracetamol 500mg",
                        "boundingBox": [10, 10, 200, 10, 200, 40, 10, 40],
                        "words": [
                            {
                                "text": "Paracetamol",
                                "boundingBox": [10, 10, 130, 10, 130, 35, 10, 35],
                            },
                            {
                                "text": "500mg",
                                "boundingBox": [140, 10, 200, 10, 200, 35, 140, 35],
                            },
                        ],
                    },
                    {
                        "text": "2 viên",
                        "boundingBox": [10, 50, 120, 50, 120, 80, 10, 80],
                        "words": [
                            {"text": "2", "boundingBox": [10, 50, 30, 50, 30, 80, 10, 80]},
                            {
                                "text": "viên",
                                "boundingBox": [40, 50, 120, 50, 120, 80, 40, 80],
                            },
                        ],
                    },
                ],
            }
        ]
    },
}


def test_azure_result_to_boxes_returns_word_boxes():
    boxes = azure_result_to_boxes(AZURE_SAMPLE)
    texts = [b["text"] for b in boxes]

    assert "Paracetamol" in texts
    assert "500mg" in texts
    assert "viên" in texts
    assert all({"l", "t", "r", "b", "w", "h", "xc", "yc"} <= set(b.keys()) for b in boxes)


def test_build_result_from_boxes_groups_lines():
    boxes = azure_result_to_boxes(AZURE_SAMPLE)
    result = build_result_from_boxes(boxes)

    assert result["blocks"]
    first_line = result["blocks"][0]["lines"][0].lower()
    assert "paracetamol" in first_line
