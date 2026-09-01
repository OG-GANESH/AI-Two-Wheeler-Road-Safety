from ultralytics import YOLO
import os

# ==========================================
# PATHS
# ==========================================

DATASET = r"D:\DriverAI\road_hazard\POTHOLE - Speed BUMP.v2i.yolov11\data.yaml"

PROJECT = r"D:\DriverAI\road_hazard\runs"

# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET):
    print("ERROR: Dataset not found!")
    print(DATASET)
    exit()

print("====================================")
print("DATASET FOUND")
print("====================================")
print(DATASET)

# ==========================================
# LOAD YOLO11 NANO
# ==========================================

model = YOLO("yolo11n.pt")

print()
print("Model loaded successfully!")
print("Classes will be taken from data.yaml.")

# ==========================================
# TRAINING
# ==========================================

results = model.train(
    data=DATASET,

    # Training
    epochs=50,

    # Image size
    imgsz=640,

    # RTX 3050 4GB
    batch=4,
    device=0,

    # Windows - safer
    workers=0,

    # Do not cache entire dataset in RAM
    cache=False,

    # Output
    project=PROJECT,
    name="pothole_speedbump_gpu",

    # Stop early if validation stops improving
    patience=10,

    # Save model
    save=True,

    # Validation
    val=True,

    # Generate plots
    plots=True,

    # Augmentation
    fliplr=0.5,
    flipud=0.0,

    # Keep deterministic
    seed=0,
)

# ==========================================
# COMPLETE
# ==========================================

print()
print("====================================")
print("TRAINING COMPLETE")
print("====================================")
print()

print("Check this folder:")
print(
    r"D:\DriverAI\road_hazard\runs\pothole_speedbump_gpu"
    r"\weights"
)

print()
print("Your best model should be:")
print(
    r"D:\DriverAI\road_hazard\runs\pothole_speedbump_gpu"
    r"\weights\best.pt"
)