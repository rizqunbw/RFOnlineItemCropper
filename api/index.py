from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
# Enable CORS for the React Dev Server
CORS(app, resources={r"/*": {"origins": "*"}})

def hex_to_hsv(hex_color):
    """Converts a hex color string to an OpenCV HSV numpy array."""
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    return hsv

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
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]
        
        # 1. Target the specific Cyan/Blue text used for item stats (e.g. #7c98c1, #8dacda, #7f9bc4)
        target_hsv_cyan = hex_to_hsv('7c98c1')
        
        # Provide a generous range around the center color to catch slight variations
        lower_cyan = np.array([max(0, target_hsv_cyan[0] - 15), max(0, target_hsv_cyan[1] - 50), max(0, target_hsv_cyan[2] - 50)])
        upper_cyan = np.array([min(179, target_hsv_cyan[0] + 15), min(255, target_hsv_cyan[1] + 50), min(255, target_hsv_cyan[2] + 50)])
        mask_cyan = cv2.inRange(hsv_img, lower_cyan, upper_cyan)
        
        # 2. Cluster the cyan text to rigidly locate the item stats block anywhere on the screen
        # Huge morphological closing to connect all cyan stat lines into one massive block
        kernel_cyan = np.ones((50, 100), np.uint8)
        cyan_closed = cv2.morphologyEx(mask_cyan, cv2.MORPH_CLOSE, kernel_cyan)
        
        cyan_contours, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_stats_block = None
        max_stats_area = 0
        
        # Find the massive stats block (ignoring tiny noise)
        for cnt in cyan_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > 2000 and area > max_stats_area:
                max_stats_area = area
                best_stats_block = {"x": x, "y": y, "w": w, "h": h}
                
        valid_candidates = []
        
        if best_stats_block:
            # 3. We found the stats block! Now we define a Region of Interest (ROI) column
            # The tooltip contains Title (top), Stats (middle), Description (bottom)
            # They all fall roughly within the same vertical column.
            
            roi_x = max(0, best_stats_block["x"] - 50) # Small horizontal padding
            roi_w = min(img_w - roi_x, best_stats_block["w"] + 100)
            
            # The title is above the stats, description is below.
            # We search a large vertical section surrounding the stats block.
            roi_y = max(0, best_stats_block["y"] - 300)
            roi_h = min(img_h - roi_y, best_stats_block["h"] + 600)
            
            roi_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(roi_mask, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), 255, -1)
            
            # 4. Target bright white text (used for titles and descriptions)
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([179, 60, 255])
            mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
            
            # Combine Cyan AND White text masks
            combined_text_mask = cv2.bitwise_or(mask_cyan, mask_white)
            
            # Only consider text WITHIN our stats column ROI
            masked_combined = cv2.bitwise_and(combined_text_mask, combined_text_mask, mask=roi_mask)
            
            # 5. Cluster all related text within the tooltip container into a final bounding box
            kernel_all_text = np.ones((100, 200), np.uint8) 
            final_closed = cv2.morphologyEx(masked_combined, cv2.MORPH_CLOSE, kernel_all_text)
            
            final_contours, _ = cv2.findContours(final_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_full_tooltip = None
            max_full_area = 0
            
            for cnt in final_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > max_full_area:
                    max_full_area = area
                    best_full_tooltip = {"x": x, "y": y, "w": w, "h": h}
                    
            if best_full_tooltip:
                # Add a comfortable padding around the text block to form the UI crop
                pad = 25
                crop_x = max(0, best_full_tooltip["x"] - pad)
                crop_y = max(0, best_full_tooltip["y"] - pad)
                crop_w = best_full_tooltip["w"] + 2*pad
                crop_h = best_full_tooltip["h"] + 2*pad
                
                # Clamp to image bounds
                crop_x = min(crop_x, img_w - 1)
                crop_y = min(crop_y, img_h - 1)
                crop_w = min(crop_w, img_w - crop_x)
                crop_h = min(crop_h, img_h - crop_y)
                
                valid_candidates.append({
                    "x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h, "area": crop_w * crop_h
                })
                
        # 6. Absolute Fallback: Original Edge Detection Logic
        # If the cyan text is completely missing from the image for some reason, try drawing a box around large edges.
        if not valid_candidates:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    area = int(w * h)
                    
                    max_allowed_area = int(float(img_h) * float(img_w) * 0.8)
                    if 50000 < area < max_allowed_area:
                        valid_candidates.append({"x": x, "y": y, "w": w, "h": h, "area": area})
                        
            # Filter out containers
            filtered_candidates = []
            def contains(rect1, rect2):
                return (rect1["x"] <= rect2["x"] and 
                        rect1["y"] <= rect2["y"] and 
                        rect1["x"] + rect1["w"] >= rect2["x"] + rect2["w"] and 
                        rect1["y"] + rect1["h"] >= rect2["y"] + rect2["h"])
                        
            for c1 in valid_candidates:
                is_container = False
                for c2 in valid_candidates:
                    if c1 != c2 and contains(c1, c2):
                        is_container = True
                        break
                if not is_container:
                    filtered_candidates.append(c1)
                    
            valid_candidates = filtered_candidates
                
        best_rect = None
        max_valid_area = 0
        
        for c in valid_candidates:
            if c["area"] > max_valid_area:
                max_valid_area = c["area"]
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
                "message": f"AI detected showcase box via text clustering"
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
