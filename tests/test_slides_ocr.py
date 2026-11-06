import cv2
import numpy as np

from lecturelens.ocr import boxes_to_lines, decide_visual_mode, near_duplicate, pick_title, postprocess
from lecturelens.slides import FrameComparer, build_runs, detect_changes


def _slide(title: str, bullets: list[str], noise: int = 0, seed: int = 0) -> np.ndarray:
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (640, 55), (120, 60, 25), -1)        # same template on every slide
    cv2.putText(img, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for i, b in enumerate(bullets):
        cv2.putText(img, b, (30, 110 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(int) + rng.integers(-noise, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def test_slide_change_detection_same_template():
    a = _slide("Gradient Descent", ["update rule", "learning rate"])
    b = _slide("Stochastic GD", ["mini batches", "noisy gradient", "epochs"])
    c = _slide("Summary", ["step size", "momentum"])
    frames = [a] * 5 + [_slide("Gradient Descent", ["update rule", "learning rate"], noise=6, seed=i) for i in range(3)] \
        + [b] * 6 + [c] * 4
    cmp = FrameComparer(12, 0.80, 0.006)
    feats = [cmp.features(f) for f in frames]
    starts = detect_changes(feats, cmp)
    assert starts == [0, 8, 14]   # compression-like noise does not create slides


def test_min_duration_merges_transitions():
    # change points at 0, 5 (1-frame transition), 6, 12 ; every_s=2 ; min 6 s => 3 frames
    runs = build_runs([0, 5, 6, 12], 16, every_s=2, min_dur_s=6)
    assert [r[0] for r in runs] == [0, 5, 12]
    assert sum(len(r) for r in runs) == 16


def _box(text, x0, y0, x1, y1, conf=0.9):
    return {"text": text, "conf": conf, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def test_boxes_to_lines_reading_order_and_confidence():
    boxes = [_box("world", 200, 100, 300, 130), _box("Hello", 50, 102, 180, 128), _box("Big Title", 40, 10, 400, 60),
             _box("@@#", 50, 200, 80, 220, conf=0.1)]
    lines = boxes_to_lines(boxes, 0.35)
    assert [l["text"] for l in lines] == ["Big Title", "Hello world"]
    assert pick_title(lines, 480) == "Big Title"


def _raw(sid, start, end, lines):
    boxes = []
    for i, (t, h) in enumerate(lines):
        y = 20 + i * 60
        boxes.append(_box(t, 20, y, 20 + 12 * len(t), y + h))
    return {"slide_id": sid, "start": start, "end": end, "thumbnail": f"t{sid}.jpg", "frame": f"f{sid}.png",
            "img_h": 480, "boxes": boxes}


def test_postprocess_headers_footers_and_near_duplicates(cfg):
    footer = ("CS101 Lecture 4", 14)
    raw = [
        _raw(0, 0, 20, [("Optimization", 40), ("minimize the loss function", 20), footer, ("1", 12)]),
        _raw(1, 20, 40, [("Update Rule", 40), ("w <- w - eta grad L", 20), footer, ("2", 12)]),
        # progressive reveal of slide 1 => merged
        _raw(2, 40, 60, [("Update Rule", 40), ("w <- w - eta grad L", 20), ("eta is the learning rate", 20), footer, ("3", 12)]),
        _raw(3, 60, 80, [("Momentum", 40), ("accelerates convergence of descent", 20), footer, ("4", 12)]),
    ]
    slides = postprocess(raw, cfg)
    assert len(slides) == 3
    assert all("CS101" not in s["text"] for s in slides)          # repeated footer removed
    assert all(not any(l["text"].isdigit() for l in s["lines"]) for s in slides)  # page numbers removed
    upd = slides[1]
    assert upd["title"] == "Update Rule" and "learning rate" in upd["text"]
    assert upd["start"] == 20 and upd["end"] == 60 and upd["merged_from"] == [1, 2]
    mode, stats = decide_visual_mode(slides, 80, cfg)
    assert mode == "high" and stats["text_slides"] == 3


def test_near_duplicate_and_low_visual_mode(cfg):
    assert near_duplicate("Gradient descent update rule", "Gradient descent update rule!", 0.9)
    assert not near_duplicate("Gradient descent", "Momentum and Adam", 0.9)
    mode, _ = decide_visual_mode([{"text": "", "start": 0, "end": 600}], 600, cfg)
    assert mode == "low"
