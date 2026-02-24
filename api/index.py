from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

TEMPLATE_SIZE = 20
TEMPLATE_THRESHOLD = 0.55
TOP_K_CANDIDATES = 80


def _load_templates():
    base_dir = Path(__file__).resolve().parent
    tl_path = base_dir / "template_tl.png"
    br_path = base_dir / "template_br.png"

    tl = cv2.imread(str(tl_path), cv2.IMREAD_GRAYSCALE)
    br = cv2.imread(str(br_path), cv2.IMREAD_GRAYSCALE)
    if tl is None or br is None:
        raise RuntimeError(
            "Template files missing. Expected api/template_tl.png and api/template_br.png."
        )
    return tl, br


TL_TEMPLATE, BR_TEMPLATE = _load_templates()


def _find_text_centroid(img_bgr):
    _, img_w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_cyan = cv2.inRange(hsv, np.array([80, 150, 170]), np.array([100, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([18, 130, 170]), np.array([42, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 55, 255]))

    text_mask = cv2.bitwise_or(mask_cyan, cv2.bitwise_or(mask_yellow, mask_white))
    text_mask = cv2.dilate(text_mask, np.ones((7, 30), np.uint8), iterations=3)

    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 100 or h < 50:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if not best:
        return None

    x, y, w, h = best
    return {"cx": x + (w // 2), "cy": y + (h // 2), "rect": best}


def _top_points(match_result, threshold=TEMPLATE_THRESHOLD, top_k=TOP_K_CANDIDATES):
    ys, xs = np.where(match_result >= threshold)
    if xs.size == 0:
        flat = match_result.ravel()
        if flat.size == 0:
            return []
        take = min(top_k, flat.size)
        idx = np.argpartition(flat, -take)[-take:]
        ys, xs = np.unravel_index(idx, match_result.shape)

    points = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        points.append((x, y, float(match_result[y, x])))

    points.sort(key=lambda p: p[2], reverse=True)
    return points[:top_k]


def _best_point(match_result):
    _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
    return {"x": int(max_loc[0]), "y": int(max_loc[1]), "score": float(max_val)}


def _box_from_corners(tl_x, tl_y, br_x, br_y):
    return {
        "x": int(tl_x),
        "y": int(tl_y),
        "w": int((br_x + TEMPLATE_SIZE) - tl_x),
        "h": int((br_y + TEMPLATE_SIZE) - tl_y),
    }


def _is_plausible_box(box, img_w, img_h):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    if x < 0 or y < 0:
        return False
    if w < 140 or h < 140:
        return False
    if w > int(img_w * 0.95) or h > int(img_h * 0.98):
        return False
    if x + w > img_w or y + h > img_h:
        return False
    return True


def detect_tooltip_box(img_bgr):
    img_h, img_w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    text_info = _find_text_centroid(img_bgr)

    tl_map = cv2.matchTemplate(gray, TL_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    br_map = cv2.matchTemplate(gray, BR_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    best_tl = _best_point(tl_map)
    best_br = _best_point(br_map)

    primary_box = _box_from_corners(
        best_tl["x"], best_tl["y"], best_br["x"], best_br["y"]
    )
    primary_ok = _is_plausible_box(primary_box, img_w, img_h)

    if primary_ok and best_tl["score"] >= 0.72 and best_br["score"] >= 0.72:
        if (best_tl["score"] + best_br["score"]) >= 1.82:
            return primary_box, "corner-argmax"
        if text_info:
            cx, cy = text_info["cx"], text_info["cy"]
            x, y, w, h = (
                primary_box["x"],
                primary_box["y"],
                primary_box["w"],
                primary_box["h"],
            )
            if x - 120 <= cx <= x + w + 120 and y - 120 <= cy <= y + h + 120:
                return primary_box, "corner-argmax"
        else:
            return primary_box, "corner-argmax"

    tl_points = _top_points(tl_map)
    br_points = _top_points(br_map)

    best_box = None
    best_score = float("-inf")

    for tl_x, tl_y, tl_score in tl_points:
        for br_x, br_y, br_score in br_points:
            w = (br_x + TEMPLATE_SIZE) - tl_x
            h = (br_y + TEMPLATE_SIZE) - tl_y

            if not (150 <= w <= int(img_w * 0.95) and 150 <= h <= int(img_h * 0.98)):
                continue
            if w * h < 55000:
                continue

            score = tl_score + br_score
            if text_info:
                cx, cy = text_info["cx"], text_info["cy"]
                in_x = tl_x - 60 <= cx <= tl_x + w + 60
                in_y = tl_y - 60 <= cy <= tl_y + h + 60
                if in_x and in_y:
                    score += 0.25
                else:
                    dx = min(abs(cx - tl_x), abs(cx - (tl_x + w)))
                    dy = min(abs(cy - tl_y), abs(cy - (tl_y + h)))
                    score -= (dx + dy) / 3000.0

            if score > best_score:
                best_score = score
                best_box = {"x": int(tl_x), "y": int(tl_y), "w": int(w), "h": int(h)}

    if best_box and _is_plausible_box(best_box, img_w, img_h):
        return best_box, "template-match"

    if text_info:
        x, y, w, h = text_info["rect"]
        pad_x = 45
        pad_y = 35
        box = {
            "x": max(0, x - pad_x),
            "y": max(0, y - pad_y),
            "w": min(img_w - max(0, x - pad_x), w + 2 * pad_x),
            "h": min(img_h - max(0, y - pad_y), h + 2 * pad_y),
        }
        return box, "text-fallback"

    box = {
        "x": int(img_w * 0.65),
        "y": int(img_h * 0.2),
        "w": int(img_w * 0.3),
        "h": int(img_h * 0.55),
    }
    return box, "default-fallback"


def to_percent_box(pixel_box, img_w, img_h):
    return {
        "unit": "%",
        "x": (float(pixel_box["x"]) / float(img_w)) * 100.0,
        "y": (float(pixel_box["y"]) / float(img_h)) * 100.0,
        "width": (float(pixel_box["w"]) / float(img_w)) * 100.0,
        "height": (float(pixel_box["h"]) / float(img_h)) * 100.0,
    }


@app.route("/detect-box", methods=["POST"])
@app.route("/api/detect-box", methods=["POST"])
def detect_box():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    img_h, img_w = img.shape[:2]
    try:
        pixel_box, method = detect_tooltip_box(img)
        pct_rect = to_percent_box(pixel_box, img_w, img_h)
        return jsonify(
            {
                "success": True,
                "box": pct_rect,
                "pixel_box": pixel_box,
                "message": f"Local CV crop success ({method})",
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal Error: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
