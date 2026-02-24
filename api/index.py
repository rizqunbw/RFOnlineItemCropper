from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
# Enable CORS for the React Dev Server
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/detect-box', methods=['POST'])
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
        
        for cnt in contours:
            # Approximate the contour to a polygon to filter out non-rectangular noise
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # The target box is rectangular
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                x, y, w, h = int(x), int(y), int(w), int(h)
                area = int(w * h)
                
                # The user's showcase box is roughly 185,000 pixels (430x432)
                # We filter out UI elements that are too small or the entire screen
                max_allowed_area = int(float(img_h) * float(img_w) * 0.8)
                if area > 50000 and area < max_allowed_area:
                    if area > max_area:
                        max_area = area
                        # Calculate percentages to prevent ReactCrop shifting on responsive screens
                        best_rect = {
                            "unit": "%",
                            "x": (float(x) / float(img_w)) * 100.0,
                            "y": (float(y) / float(img_h)) * 100.0,
                            "width": (float(w) / float(img_w)) * 100.0,
                            "height": (float(h) / float(img_h)) * 100.0
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
