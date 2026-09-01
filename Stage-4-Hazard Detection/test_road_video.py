from ultralytics import YOLO
import os

# ==============================
# MODEL
# ==============================
MODEL = r"D:\DriverAI\road_hazard\runs\pothole_speedbump_gpu-3\weights\best.pt"

# ==============================
# INPUT VIDEO
# ==============================
VIDEO = r"D:\DriverAI\road_hazard\breaker.mp4"

# ==============================
# OUTPUT
# ==============================
OUTPUT_DIR = r"D:\DriverAI\road_hazard\video_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = YOLO(MODEL)

print("Model loaded!")
print("Classes:", model.names)

print("\nStarting video detection...")

results = model.predict(
    source=VIDEO,
    conf=0.50,
    imgsz=640,
    device=0,
    save=True,
    project=OUTPUT_DIR,
    name="road_test",
    exist_ok=True,
    verbose=True
)

print("\n================================")
print("VIDEO DETECTION COMPLETE")
print("================================")
print(f"Output folder: {OUTPUT_DIR}\\road_test")