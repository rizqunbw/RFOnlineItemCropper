from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
# Enable CORS for the React Dev Server
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/detect-box', methods=['POST'])
@app.route('/api/detect-box', methods=['POST'])
def detect_box():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    
    # Read image directly from memory into OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    try:
        # Algorithm: Canny Edge Detection + Contour Area Heuristics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        shape = img.shape
        img_h, img_w = int(shape[0]), int(shape[1])
        
        candidates = []
        for cnt in contours:
            # Approximate the contour to a polygon to filter out non-rectangular noise
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # The target box is roughly rectangular, but corners can add vertices
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                x, y, w, h = int(x), int(y), int(w), int(h)
                area = int(w * h)
                
                # The user's showcase box is quite large.
                # We filter out UI elements that are too small or the entire screen
                max_allowed_area = int(float(img_h) * float(img_w) * 0.8)
                if area > 50000 and area < max_allowed_area:
                    candidates.append({"x": x, "y": y, "w": w, "h": h, "area": area})
                    
        # Try to find the [Description] text template to perfectly isolate the item box
        desc_box = None
        template_path = os.path.join(os.path.dirname(__file__), 'desc_template.png')
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
            if template is not None:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.8)
                desc_points = list(zip(*loc[::-1]))
                if desc_points:
                    th, tw = template_gray.shape
                    desc_box = {"x": int(desc_points[0][0]), "y": int(desc_points[0][1]), "w": tw, "h": th}

        # Filter out candidates that completely contain another candidate
        # This prevents picking a large macro-UI panel instead of the inner item tooltip.
        def contains(rect1, rect2):
            return (rect1["x"] <= rect2["x"] and 
                    rect1["y"] <= rect2["y"] and 
                    rect1["x"] + rect1["w"] >= rect2["x"] + rect2["w"] and 
                    rect1["y"] + rect1["h"] >= rect2["y"] + rect2["h"])
                    
        valid_candidates = []
        
        if desc_box:
            # If we found the [Description] text, the correct box is the one that contains it.
            # We want the SMALLEST box that contains the description.
            containing_candidates = [c for c in candidates if contains(c, desc_box)]
            if containing_candidates:
                # Add only the smallest containing box
                smallest_container = min(containing_candidates, key=lambda c: c["area"])
                valid_candidates.append(smallest_container)
        
        if not valid_candidates:
            # Fallback to the old logic if description text wasn't found or matched no boxes
            for c1 in candidates:
                is_container = False
                for c2 in candidates:
                    if c1 != c2 and contains(c1, c2):
                        is_container = True
                        break
                if not is_container:
                    valid_candidates.append(c1)
                
        for c in valid_candidates:
            if c["area"] > max_area:
                max_area = c["area"]
                # Calculate percentages to prevent ReactCrop shifting on responsive screens
                best_rect = {
                    "unit": "%",
                    "x": (float(c["x"]) / float(img_w)) * 100.0,
                    "y": (float(c["y"]) / float(img_h)) * 100.0,
                    "width": (float(c["w"]) / float(img_w)) * 100.0,
                    "height": (float(c["h"]) / float(img_h)) * 100.0
                }
        
        if best_rect:
            return jsonify({
                "success": True,
                "box": best_rect,
                "message": f"AI detected showcase box"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Could not confidently identify the rectangular item box.",
                "box": None
            }), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
