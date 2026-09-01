from ultralytics import YOLO
import os

# ==========================================
# TRAINED MODEL
# ==========================================

MODEL = r"D:\DriverAI\road_hazard\runs\pothole_speedbump_gpu-3\weights\best.pt"

# ==========================================
# TEST IMAGE
# ==========================================

SOURCE = r"D:\DriverAI\road_hazard\test.png"

# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL):
    print("ERROR: Model not found!")
    print(MODEL)
    exit()

if not os.path.exists(SOURCE):
    print("ERROR: Test image not found!")
    print(SOURCE)
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = YOLO(MODEL)

print("Model loaded successfully!")
print("Classes:", model.names)

# ==========================================
# DETECTION
# ==========================================

print("\nStarting detection...")

results = model.predict(
    source=SOURCE,
    conf=0.50,
    imgsz=640,
    device=0,
    save=True,
    project=r"D:\DriverAI\road_hazard\test_results",
    name="hazard_test",
    exist_ok=True
)

# ==========================================
# COMPLETE
# ==========================================

print("\n====================================")
print("DETECTION COMPLETE")
print("====================================")

print("Result saved to:")
print(r"D:\DriverAI\road_hazard\test_results\hazard_test")