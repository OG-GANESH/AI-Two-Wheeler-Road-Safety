from ultralytics import YOLO
import cv2


# ============================================
# LOAD YOLO MODEL
# ============================================

model = YOLO("yolo11n.pt")


# ============================================
# OPEN CAMERA
# ============================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================
# CAMERA RESOLUTION
# ============================================

WIDTH = 640
HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ============================================
# RIDER PATH
# ============================================

PATH_LEFT = 160
PATH_RIGHT = 480


# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break


    # ========================================
    # YOLO DETECTION
    # ========================================

    results = model(frame, verbose=False)

    result = results[0]


    # ========================================
    # DRAW RIDER PATH
    # ========================================

    cv2.rectangle(
        frame,
        (PATH_LEFT, 0),
        (PATH_RIGHT, HEIGHT),
        (255, 255, 255),
        2
    )


    # ========================================
    # PROCESS DETECTED OBJECTS
    # ========================================

    if result.boxes is not None:

        for box in result.boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # Confidence
            confidence = float(box.conf[0])


            # Class
            class_id = int(box.cls[0])
            class_name = result.names[class_id]


            # =================================
            # OBJECT CENTER
            # =================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # =================================
            # CHECK RIDER PATH
            # =================================

            in_rider_path = (
                PATH_LEFT <= center_x <= PATH_RIGHT
            )


            # =================================
            # PRINT INFORMATION
            # =================================

            print(
                f"Object: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Center: ({center_x}, {center_y}) | "
                f"In Path: {in_rider_path}"
            )


            # =================================
            # DRAW INFORMATION
            # =================================

            if in_rider_path:

                cv2.putText(
                    frame,
                    "IN RIDER PATH",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "OUTSIDE PATH",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


            # Draw bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Label

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


            # Center point

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


    # ========================================
    # DISPLAY
    # ========================================

    cv2.imshow(
        "Stage 2.3 - Rider Path Detection",
        frame
    )


    # ========================================
    # EXIT
    # ========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()