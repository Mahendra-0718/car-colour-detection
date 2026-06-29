import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text
from PIL import Image, ImageTk
from ultralytics import YOLO

# ── Load YOLO model
model = YOLO("yolov8n.pt")


def preprocess_crop(crop):
    """Mild blur to reduce reflection/shadow noise."""
    return cv2.GaussianBlur(crop, (5, 5), 0)


def get_dominant_colour(car_crop):
    """
    Robust colour detection tuned for AERIAL / side-view traffic images.
    Key insight: aerial car roofs have LOW saturation due to direct sunlight,
    so all saturation thresholds must be kept very low.
    Never returns 'Unknown'.
    """
    if car_crop is None or car_crop.size == 0:
        return "Grey"

    h, w = car_crop.shape[:2]
    if h < 10 or w < 10:
        return "Grey"

    # Use middle 60% of crop (avoids road/background bleed on edges)
    body = car_crop[int(h * 0.20):int(h * 0.80), int(w * 0.15):int(w * 0.85)]
    if body.size == 0:
        body = car_crop  # fallback to full crop

    body         = preprocess_crop(body)
    hsv          = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    total_pixels = body.shape[0] * body.shape[1]

    # ── Helper: ratio of pixels matching a mask
    def ratio(mask):
        return cv2.countNonZero(mask) / total_pixels

    # ────────────────────────────────────────────────
    # STEP 1: Black  (Value very dark)
    # ────────────────────────────────────────────────
    black_mask = cv2.inRange(hsv,
                    np.array([0,   0,   0]),
                    np.array([180, 255, 55]))
    if ratio(black_mask) > 0.50:
        return "Black"

    # ────────────────────────────────────────────────
    # STEP 2: White  (Saturation very low, Value high)
    # ────────────────────────────────────────────────
    white_mask = cv2.inRange(hsv,
                    np.array([0,   0,  185]),
                    np.array([180, 35, 255]))
    if ratio(white_mask) > 0.40:
        return "White"

    # ────────────────────────────────────────────────
    # STEP 3: Silver / Grey
    #   Low saturation (0-55) + medium brightness (55-200)
    #   Checked BEFORE colours so it can be beaten by them
    # ────────────────────────────────────────────────
    silver_mask = cv2.inRange(hsv,
                    np.array([0,   0,  55]),
                    np.array([180, 55, 200]))
    silver_ratio = ratio(silver_mask)

    # ────────────────────────────────────────────────
    # STEP 4: Colour ranges — LOW saturation minimums
    #   (aerial/sunny roofs: S can be as low as 20)
    # ────────────────────────────────────────────────
    colour_ranges = {
        "Red": [
            (np.array([0,   30, 40]),  np.array([12,  255, 255])),
            (np.array([158, 30, 40]),  np.array([180, 255, 255])),
        ],
        "Orange": [
            (np.array([13,  30, 50]),  np.array([22,  255, 255])),
        ],
        "Yellow": [
            (np.array([23,  25, 60]),  np.array([38,  255, 255])),
        ],
        "Green": [
            (np.array([39,  20, 25]),  np.array([90,  255, 255])),
        ],
        "Blue": [
            (np.array([91,  25, 25]),  np.array([135, 255, 255])),
        ],
        "Purple": [
            (np.array([125, 20, 25]),  np.array([158, 255, 255])),
        ],
        "Brown": [
            # warm hue, moderate sat, dark value
            (np.array([8,   40, 20]),  np.array([20,  200, 140])),
        ],
        "Maroon": [
            (np.array([0,   50, 20]),  np.array([10,  255,  90])),
            (np.array([158, 50, 20]),  np.array([180, 255,  90])),
        ],
        "Beige": [
            # Very low saturation warm tone (common on light-coloured cars)
            (np.array([15,  10, 160]), np.array([35,  60,  255])),
        ],
    }

    pixel_counts = {}
    for name, ranges in colour_ranges.items():
        count = 0
        for (lo, hi) in ranges:
            count += cv2.countNonZero(cv2.inRange(hsv, lo, hi))
        pixel_counts[name] = count

    best_colour = max(pixel_counts, key=pixel_counts.get)
    best_count  = pixel_counts[best_colour]
    best_ratio  = best_count / total_pixels

    # ── Decision logic:
    # If a colour wins clearly → return it
    # If silver/grey is dominant and beats the colour → Grey
    # Otherwise force the best colour (no Unknown ever)

    # Colour wins if it covers ≥ 8% of body pixels
    if best_ratio >= 0.08:
        # Extra check: if silver is MUCH stronger, prefer Grey
        if silver_ratio > best_ratio * 2.5 and silver_ratio > 0.35:
            return "Silver/Grey"
        return best_colour

    # Colour too weak — use Grey if it has enough coverage
    if silver_ratio >= 0.25:
        return "Silver/Grey"

    # Last resort: use Black or White based on average brightness
    avg_v = float(np.mean(hsv[:, :, 2]))
    if avg_v < 80:
        return "Black"
    if avg_v > 170:
        return "White"

    # Absolute fallback — return the colour with most pixels (never Unknown)
    return best_colour if best_count > 0 else "Silver/Grey"


# ── Main detection function
def detect_cars(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, {}, 0

    results       = model(frame, verbose=False)[0]
    colour_counts = {}
    car_count     = 0
    output_frame  = frame.copy()

    # Colour → BGR for bounding boxes
    COLOUR_BGR = {
        "Red"        : (0,   0,   220),
        "Orange"     : (0,   140, 255),
        "Yellow"     : (0,   220, 220),
        "Green"      : (0,   180,   0),
        "Blue"       : (220,   0,   0),
        "Purple"     : (180,   0, 180),
        "Brown"      : (30,   60, 100),
        "Maroon"     : (0,    0,  128),
        "Beige"      : (150, 200, 220),
        "Black"      : (80,  80,  80),
        "White"      : (220, 220, 220),
        "Silver/Grey": (170, 170, 170),
    }

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        label      = model.names[cls_id]
        confidence = float(box.conf[0])

        if label not in ["car", "truck", "bus", "motorcycle"]:
            continue
        if confidence < 0.25:   # lower threshold to catch more vehicles
            continue

        car_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp to frame boundaries
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0:
            continue

        colour_name = get_dominant_colour(car_crop)
        colour_counts[colour_name] = colour_counts.get(colour_name, 0) + 1

        box_colour = COLOUR_BGR.get(colour_name, (0, 255, 255))
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_colour, 2)
        cv2.putText(
            output_frame,
            f"{colour_name} ({confidence:.0%})",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_colour, 2
        )

    return output_frame, colour_counts, car_count


# ── GUI
class CarColourApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Colour Detection")
        self.root.geometry("950x700")
        self.root.configure(bg="#1e1e2e")

        Label(root,
              text="Car Colour Detection at Traffic Signal",
              font=("Arial", 16, "bold"),
              bg="#1e1e2e", fg="white").pack(pady=10)

        btn_frame = Frame(root, bg="#1e1e2e")
        btn_frame.pack()

        Button(btn_frame,
               text="📂 Upload Image",
               command=self.upload_image,
               bg="#7c3aed", fg="white",
               font=("Arial", 12), padx=10).pack(side="left", padx=10)

        Button(btn_frame,
               text="🎥 Use Webcam",
               command=self.use_webcam,
               bg="#0891b2", fg="white",
               font=("Arial", 12), padx=10).pack(side="left", padx=10)

        self.image_label = Label(root, bg="#1e1e2e")
        self.image_label.pack(pady=10)

        result_frame = Frame(root, bg="#1e1e2e")
        result_frame.pack()

        Label(result_frame,
              text="Detection Results:",
              font=("Arial", 13, "bold"),
              bg="#1e1e2e", fg="#a3e635").pack()

        self.result_text = Text(result_frame,
                                height=8, width=60,
                                bg="#2e2e3e", fg="white",
                                font=("Courier", 11), relief="flat")
        self.result_text.pack(pady=5)

    def show_image(self, cv_image):
        rgb     = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((700, 380))
        tk_img  = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    def show_results(self, colour_counts, car_count):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END,
            f"🚗 Total Vehicles Detected : {car_count}\n")
        self.result_text.insert(tk.END,
            f"👥 People at Signal (est.)  : {car_count} person(s)\n\n")
        self.result_text.insert(tk.END, "── Colour Breakdown ──\n")
        for colour, count in sorted(colour_counts.items(),
                                    key=lambda x: -x[1]):
            bar = "█" * count
            self.result_text.insert(
                tk.END, f"  {colour:<12} : {bar} ({count})\n")

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        output_frame, colour_counts, car_count = detect_cars(path)
        if output_frame is None:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "❌ Could not read image.")
            return
        self.show_image(output_frame)
        self.show_results(colour_counts, car_count)

    def use_webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "❌ Webcam not found.")
            return
        temp_path = "temp_webcam.jpg"
        cv2.imwrite(temp_path, frame)
        output_frame, colour_counts, car_count = detect_cars(temp_path)
        self.show_image(output_frame)
        self.show_results(colour_counts, car_count)


# ── Run
if __name__ == "__main__":
    root = tk.Tk()
    app  = CarColourApp(root)
    root.mainloop()