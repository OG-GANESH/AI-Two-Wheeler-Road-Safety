# ============================================================
# AI-BASED RIDER MONITORING SYSTEM
# STAGE 1 - COMPLETE PROTOTYPE
#
# Features:
#   1. Face detection
#   2. EAR calculation
#   3. Blink detection
#   4. Drowsiness detection
#   5. MAR calculation
#   6. Yawning detection
#   7. Head pose estimation
#   8. Left/right distraction detection
#   9. Head-down detection
#  10. Duration-based filtering
#
# ============================================================


import cv2
import mediapipe as mp
import math
import time
import os
import numpy as np
import serial


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# MediaPipe model
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)


# ------------------------------------------------------------
# DROWSINESS SETTINGS
# ------------------------------------------------------------

# Your calibration:
#
# Eyes open  -> approximately 0.27 - 0.38
# Blink      -> approximately 0.18
# Eyes closed -> approximately 0.02 - 0.05
#
# Therefore 0.15 is a good starting threshold.

EAR_THRESHOLD = 0.15

# Eyes continuously closed for this duration
# => DROWSY

DROWSY_TIME = 1.5


# ------------------------------------------------------------
# EAR SMOOTHING
# ------------------------------------------------------------

ALPHA = 0.4


# ------------------------------------------------------------
# YAWNING SETTINGS
# ------------------------------------------------------------

# Your calibration:
#
# Mouth closed -> 0.238 - 0.267
# Talking      -> 0.484 - 0.550
# Mouth open   -> 0.620 - 0.690
# Yawn         -> 1.077 - 1.200
#
# Therefore 0.90 is a reasonable starting threshold.

MAR_THRESHOLD = 0.90

YAWN_TIME = 0.8


# ------------------------------------------------------------
# DISTRACTION SETTINGS
# ------------------------------------------------------------

# Your calibration:
#
# Forward:
# approximately -2° to +3°
#
# Left:
# +27.2°
#
# Right:
# -50°
#
# Therefore 20° provides a useful separation.

YAW_THRESHOLD = 20.0

# Head must remain turned for this long
# before declaring distraction.

DISTRACTION_TIME = 1.5


# ------------------------------------------------------------
# HEAD DOWN SETTINGS
# ------------------------------------------------------------

# Your current calibration:
#
# Forward pitch ≈ -1.5 to +3.6
# Down pitch    ≈ +7.5
#
# This is NOT fully calibrated yet.
#
# We therefore use a conservative preliminary
# threshold.

DOWN_PITCH_THRESHOLD = 3.0

DOWN_TIME = 1.5


# ============================================================
# ESP32 SERIAL CONNECTION
# ============================================================

ESP32_PORT = "COM3"
ESP32_BAUDRATE = 115200

try:
    esp32 = serial.Serial(ESP32_PORT, ESP32_BAUDRATE, timeout=0.1)
    time.sleep(2)
    print(f"ESP32 connected on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"WARNING: ESP32 connection failed on {ESP32_PORT}: {e}")
    esp32 = None

last_esp32_command = None


def send_to_esp32(command):
    """Send a command to ESP32 only when the command changes."""
    global last_esp32_command

    if command == last_esp32_command:
        return

    if esp32 is not None and esp32.is_open:
        try:
            esp32.write((command + "\n").encode("utf-8"))
            print(f"[ESP32] Sent: {command}")

            # Read optional ACK from ESP32
            time.sleep(0.01)
            while esp32.in_waiting:
                response = esp32.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    print(f"[ESP32] {response}")

        except serial.SerialException as e:
            print(f"[ESP32] Serial error: {e}")

    last_esp32_command = command


# ============================================================
# 2. MEDIAPIPE FACE LANDMARKER
# ============================================================

BaseOptions = python.BaseOptions


options = vision.FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=vision.RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# 3. FACIAL LANDMARK INDICES
# ============================================================


# ------------------------------------------------------------
# LEFT EYE
# ------------------------------------------------------------

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


# ------------------------------------------------------------
# RIGHT EYE
# ------------------------------------------------------------

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ------------------------------------------------------------
# MOUTH
# ------------------------------------------------------------

MOUTH = [
    61,
    13,
    14,
    291,
    0,
    17
]


# ------------------------------------------------------------
# HEAD POSE
# ------------------------------------------------------------

HEAD_POSE_POINTS = [

    1,       # Nose
    152,     # Chin
    33,      # Left eye
    263,     # Right eye
    61,      # Left mouth
    291      # Right mouth

]


# ============================================================
# 4. 3D FACE MODEL
# ============================================================

FACE_3D_MODEL = np.array([

    (0.0, 0.0, 0.0),           # Nose

    (0.0, -330.0, -65.0),      # Chin

    (-225.0, 170.0, -135.0),   # Left eye

    (225.0, 170.0, -135.0),    # Right eye

    (-150.0, -150.0, -125.0),  # Left mouth

    (150.0, -150.0, -125.0)    # Right mouth

], dtype=np.float64)


# ============================================================
# 5. DISTANCE FUNCTION
# ============================================================

def euclidean_distance(point1, point2):

    x1, y1 = point1

    x2, y2 = point2

    return math.sqrt(

        (x2 - x1) ** 2

        +

        (y2 - y1) ** 2

    )


# ============================================================
# 6. EAR CALCULATION
# ============================================================

def calculate_ear(eye_points):

    vertical_1 = euclidean_distance(

        eye_points[1],

        eye_points[5]

    )


    vertical_2 = euclidean_distance(

        eye_points[2],

        eye_points[4]

    )


    horizontal = euclidean_distance(

        eye_points[0],

        eye_points[3]

    )


    if horizontal == 0:

        return 0.0


    ear = (

        vertical_1 +

        vertical_2

    ) / (

        2.0 * horizontal

    )


    return ear


# ============================================================
# 7. MAR CALCULATION
# ============================================================

def calculate_mar(mouth_points):

    vertical_1 = euclidean_distance(

        mouth_points[1],

        mouth_points[5]

    )


    vertical_2 = euclidean_distance(

        mouth_points[2],

        mouth_points[4]

    )


    horizontal = euclidean_distance(

        mouth_points[0],

        mouth_points[3]

    )


    if horizontal == 0:

        return 0.0


    mar = (

        vertical_1 +

        vertical_2

    ) / (

        2.0 * horizontal

    )


    return mar


# ============================================================
# 8. LANDMARK → PIXEL COORDINATES
# ============================================================

def landmark_to_pixel(
        landmark,
        width,
        height):

    x = int(

        landmark.x * width

    )

    y = int(

        landmark.y * height

    )

    return x, y


# ============================================================
# 9. GET LANDMARK POINTS
# ============================================================

def get_points(
        frame,
        landmarks,
        indices):

    height, width, _ = frame.shape

    points = []


    for index in indices:

        landmark = landmarks[index]

        point = landmark_to_pixel(

            landmark,

            width,

            height

        )

        points.append(point)


    return points


# ============================================================
# 10. DRAW LANDMARKS
# ============================================================

def draw_points(
        frame,
        points):

    for point in points:

        cv2.circle(

            frame,

            point,

            2,

            (0, 255, 0),

            -1

        )


    for i in range(len(points)):

        start = points[i]

        end = points[
            (i + 1) % len(points)
        ]


        cv2.line(

            frame,

            start,

            end,

            (255, 255, 0),

            1

        )


# ============================================================
# 11. ANGLE NORMALIZATION
# ============================================================

def normalize_angle(angle):

    if angle > 90:

        angle -= 180


    elif angle < -90:

        angle += 180


    return angle


# ============================================================
# 12. HEAD POSE ESTIMATION
# ============================================================

def estimate_head_pose(
        frame,
        landmarks):

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Get 2D facial points
    # --------------------------------------------------------

    image_points = []


    for index in HEAD_POSE_POINTS:

        landmark = landmarks[index]


        x = landmark.x * width

        y = landmark.y * height


        image_points.append(

            (x, y)

        )


    image_points = np.array(

        image_points,

        dtype=np.float64

    )


    # --------------------------------------------------------
    # Camera matrix
    # --------------------------------------------------------

    focal_length = width


    center = (

        width / 2,

        height / 2

    )


    camera_matrix = np.array([

        [

            focal_length,

            0,

            center[0]

        ],

        [

            0,

            focal_length,

            center[1]

        ],

        [

            0,

            0,

            1

        ]

    ], dtype=np.float64)


    # --------------------------------------------------------
    # Camera distortion
    # --------------------------------------------------------

    dist_coeffs = np.zeros(

        (4, 1),

        dtype=np.float64

    )


    # --------------------------------------------------------
    # Solve PnP
    # --------------------------------------------------------

    success, rotation_vector, translation_vector = (

        cv2.solvePnP(

            FACE_3D_MODEL,

            image_points,

            camera_matrix,

            dist_coeffs,

            flags=cv2.SOLVEPNP_ITERATIVE

        )

    )


    if not success:

        return None, None, None


    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    rotation_matrix, _ = cv2.Rodrigues(

        rotation_vector

    )


    # --------------------------------------------------------
    # Euler angles
    # --------------------------------------------------------

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(

        rotation_matrix

    )


    pitch = normalize_angle(

        angles[0]

    )


    yaw = normalize_angle(

        angles[1]

    )


    roll = normalize_angle(

        angles[2]

    )


    return pitch, yaw, roll


# ============================================================
# 13. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    landmarker.close()

    exit()


# ============================================================
# 14. CAMERA SETTINGS
# ============================================================

# You can change these later.

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# 15. VARIABLES
# ============================================================


# ------------------------------------------------------------
# FPS
# ------------------------------------------------------------

previous_time = time.time()


# ------------------------------------------------------------
# MEDIAPIPE TIMESTAMP
# ------------------------------------------------------------

last_timestamp_ms = 0


# ------------------------------------------------------------
# EAR
# ------------------------------------------------------------

smoothed_ear = None

eyes_closed_start = None

blink_count = 0

closed_duration = 0.0


# ------------------------------------------------------------
# MAR
# ------------------------------------------------------------

yawn_start = None

yawn_count = 0

yawn_duration = 0.0

yawn_detected = False


# ------------------------------------------------------------
# HEAD POSE
# ------------------------------------------------------------

pitch = 0.0

yaw = 0.0

roll = 0.0


# ------------------------------------------------------------
# DISTRACTION
# ------------------------------------------------------------

distraction_start = None

distraction_duration = 0.0

distraction_detected = False

distraction_direction = "FORWARD"


# ------------------------------------------------------------
# DOWNWARD LOOK
# ------------------------------------------------------------

down_start = None

down_duration = 0.0

down_detected = False


# ------------------------------------------------------------
# MAIN STATUS
# ------------------------------------------------------------

status = "ALERT"


# ============================================================
# 16. MAIN LOOP
# ============================================================

while True:


    # ========================================================
    # READ CAMERA
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    # ========================================================
    # MIRROR IMAGE
    # ========================================================

    frame = cv2.flip(

        frame,

        1

    )


    # ========================================================
    # FRAME SIZE
    # ========================================================

    height, width, _ = frame.shape


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()


    delta_time = (

        current_time

        -

        previous_time

    )


    if delta_time > 0:

        fps = 1.0 / delta_time

    else:

        fps = 0.0


    previous_time = current_time


    # ========================================================
    # BGR → RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame

    )


    # ========================================================
    # MONOTONIC TIMESTAMP
    # ========================================================
    #
    # IMPORTANT:
    # MediaPipe VIDEO mode requires timestamps to ALWAYS
    # increase.
    #
    # We use system monotonic time and guarantee that the
    # timestamp is larger than the previous one.
    #
    # ========================================================

    timestamp_ms = int(

        time.monotonic() * 1000

    )


    if timestamp_ms <= last_timestamp_ms:

        timestamp_ms = (

            last_timestamp_ms + 1

        )


    last_timestamp_ms = timestamp_ms


    # ========================================================
    # FACE LANDMARK DETECTION
    # ========================================================

    result = landmarker.detect_for_video(

        mp_image,

        timestamp_ms

    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    left_ear = 0.0

    right_ear = 0.0

    average_ear = 0.0

    mouth_mar = 0.0

    closed_duration = 0.0

    yawn_duration = 0.0


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if result.face_landmarks:


        face_landmarks = (

            result.face_landmarks[0]

        )


        # ====================================================
        # EYE POINTS
        # ====================================================

        left_eye_points = get_points(

            frame,

            face_landmarks,

            LEFT_EYE

        )


        right_eye_points = get_points(

            frame,

            face_landmarks,

            RIGHT_EYE

        )


        # ====================================================
        # MOUTH POINTS
        # ====================================================

        mouth_points = get_points(

            frame,

            face_landmarks,

            MOUTH

        )


        # ====================================================
        # DRAW EYE LANDMARKS
        # ====================================================

        draw_points(

            frame,

            left_eye_points

        )


        draw_points(

            frame,

            right_eye_points

        )


        # ====================================================
        # DRAW MOUTH LANDMARKS
        # ====================================================

        draw_points(

            frame,

            mouth_points

        )


        # ====================================================
        # EAR
        # ====================================================

        left_ear = calculate_ear(

            left_eye_points

        )


        right_ear = calculate_ear(

            right_eye_points

        )


        average_ear = (

            left_ear +

            right_ear

        ) / 2.0


        # ====================================================
        # EAR SMOOTHING
        # ====================================================

        if smoothed_ear is None:

            smoothed_ear = average_ear


        else:

            smoothed_ear = (

                ALPHA * average_ear

                +

                (1 - ALPHA)

                * smoothed_ear

            )


        # ====================================================
        # EYE STATE
        # ====================================================

        eyes_closed = (

            smoothed_ear

            <

            EAR_THRESHOLD

        )


        # ====================================================
        # EYES CLOSED
        # ====================================================

        if eyes_closed:


            if eyes_closed_start is None:

                eyes_closed_start = (

                    time.time()

                )


            closed_duration = (

                time.time()

                -

                eyes_closed_start

            )


        # ====================================================
        # EYES OPEN
        # ====================================================

        else:


            # ------------------------------------------------
            # Detect blink
            # ------------------------------------------------

            if eyes_closed_start is not None:


                closure_duration = (

                    time.time()

                    -

                    eyes_closed_start

                )


                if (

                    closure_duration >= 0.05

                    and

                    closure_duration < DROWSY_TIME

                ):

                    blink_count += 1


            eyes_closed_start = None

            closed_duration = 0.0


        # ====================================================
        # MAR
        # ====================================================

        mouth_mar = calculate_mar(

            mouth_points

        )


        # ====================================================
        # YAWNING
        # ====================================================

        if mouth_mar > MAR_THRESHOLD:


            if yawn_start is None:

                yawn_start = (

                    time.time()

                )


            yawn_duration = (

                time.time()

                -

                yawn_start

            )


            if (

                yawn_duration >= YAWN_TIME

            ):


                if not yawn_detected:

                    yawn_count += 1


                yawn_detected = True


        else:


            yawn_start = None

            yawn_duration = 0.0

            yawn_detected = False


        # ====================================================
        # HEAD POSE
        # ====================================================

        pitch, yaw, roll = (

            estimate_head_pose(

                frame,

                face_landmarks

            )

        )


        # ====================================================
        # HEAD POSE VALID
        # ====================================================

        if yaw is not None:


            # =================================================
            # LEFT / RIGHT DISTRACTION
            # =================================================

            if yaw > YAW_THRESHOLD:

                distraction_direction = "LEFT"

                head_turned = True


            elif yaw < -YAW_THRESHOLD:

                distraction_direction = "RIGHT"

                head_turned = True


            else:

                distraction_direction = "FORWARD"

                head_turned = False


            # =================================================
            # DISTRACTION TIMER
            # =================================================

            if head_turned:


                if distraction_start is None:

                    distraction_start = (

                        time.time()

                    )


                distraction_duration = (

                    time.time()

                    -

                    distraction_start

                )


                if (

                    distraction_duration

                    >=

                    DISTRACTION_TIME

                ):

                    distraction_detected = True


            else:


                distraction_start = None

                distraction_duration = 0.0

                distraction_detected = False


            # =================================================
            # HEAD DOWN DETECTION
            # =================================================

            if pitch > DOWN_PITCH_THRESHOLD:


                if down_start is None:

                    down_start = (

                        time.time()

                    )


                down_duration = (

                    time.time()

                    -

                    down_start

                )


                if (

                    down_duration

                    >=

                    DOWN_TIME

                ):

                    down_detected = True


            else:


                down_start = None

                down_duration = 0.0

                down_detected = False


        else:


            distraction_start = None

            distraction_duration = 0.0

            distraction_detected = False


            down_start = None

            down_duration = 0.0

            down_detected = False


    # ========================================================
    # NO FACE
    # ========================================================

    else:


        status = "NO FACE"


        eyes_closed_start = None

        yawn_start = None

        distraction_start = None

        down_start = None


        smoothed_ear = None


        pitch = 0.0

        yaw = 0.0

        roll = 0.0


        distraction_duration = 0.0

        down_duration = 0.0


        distraction_detected = False

        down_detected = False


    # ========================================================
    # DETERMINE MAIN RIDER STATUS
    # ========================================================

    if not result.face_landmarks:

        status = "NO FACE"


    elif (

        smoothed_ear is not None

        and

        closed_duration >= DROWSY_TIME

    ):

        status = "DROWSY"


    elif distraction_detected:

        status = "DISTRACTED"


    elif down_detected:

        status = "POSSIBLE PHONE"


    elif (

        smoothed_ear is not None

        and

        smoothed_ear < EAR_THRESHOLD

    ):

        status = "EYES CLOSED"


    else:

        status = "ALERT"


    # ========================================================
    # ESP32 OUTPUT
    # ========================================================
    # The computer's detected rider status is converted into a
    # simple serial command for the ESP32.
    #
    # ALERT          -> LED OFF
    # EYES CLOSED    -> LED BLINK
    # DROWSY         -> LED FAST BLINK
    # DISTRACTED     -> LED BLINK
    # POSSIBLE PHONE -> LED BLINK
    # NO FACE        -> LED OFF

    if status == "DROWSY":
        esp32_command = "DROWSY"

    elif status == "DISTRACTED":
        esp32_command = "DISTRACTED"

    elif status == "POSSIBLE PHONE":
        esp32_command = "PHONE"

    elif status == "EYES CLOSED":
        esp32_command = "EYES_CLOSED"

    elif status == "NO FACE":
        esp32_command = "NO_FACE"

    else:
        esp32_command = "ALERT"

    send_to_esp32(esp32_command)


    # Print current status in terminal only when it changes.
    # This keeps the terminal readable instead of printing every frame.
    if not hasattr(send_to_esp32, "last_status"):
        send_to_esp32.last_status = None

    if status != send_to_esp32.last_status:
        print(f"[RIDER STATUS] {status}")
        send_to_esp32.last_status = status


    # ========================================================
    # DASHBOARD
    # ========================================================


    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if result.face_landmarks:

        cv2.putText(

            frame,

            "FACE: DETECTED",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2

        )

    else:

        cv2.putText(

            frame,

            "FACE: NOT DETECTED",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 0, 255),

            2

        )


    # --------------------------------------------------------
    # EAR
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Left EAR: {left_ear:.3f}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Right EAR: {right_ear:.3f}",

        (20, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # SMOOTHED EAR
    # --------------------------------------------------------

    display_ear = (

        smoothed_ear

        if smoothed_ear is not None

        else 0.0

    )


    cv2.putText(

        frame,

        f"Smoothed EAR: {display_ear:.3f}",

        (20, 130),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # EAR THRESHOLD
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"EAR Threshold: {EAR_THRESHOLD:.2f}",

        (20, 160),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # EYE CLOSED TIME
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Eye Closed: {closed_duration:.2f} s",

        (20, 190),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # BLINK COUNT
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Blinks: {blink_count}",

        (20, 220),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 250),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 0),

        2

    )


    # ========================================================
    # MOUTH SECTION
    # ========================================================

    cv2.putText(

        frame,

        f"MAR: {mouth_mar:.3f}",

        (20, 300),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"MAR Threshold: {MAR_THRESHOLD:.2f}",

        (20, 330),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Yawn Time: {yawn_duration:.2f} s",

        (20, 360),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Yawns: {yawn_count}",

        (20, 390),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # MOUTH STATUS
    # --------------------------------------------------------

    if yawn_detected:

        mouth_status = "YAWNING"

        mouth_color = (0, 165, 255)

    else:

        mouth_status = "NORMAL"

        mouth_color = (255, 255, 255)


    cv2.putText(

        frame,

        f"MOUTH: {mouth_status}",

        (20, 425),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        mouth_color,

        2

    )


    # ========================================================
    # HEAD POSE SECTION
    # ========================================================

    cv2.putText(

        frame,

        f"Pitch: {pitch:.1f}",

        (width - 230, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Yaw: {yaw:.1f}",

        (width - 230, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"Roll: {roll:.1f}",

        (width - 230, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # HEAD DIRECTION
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Head: {distraction_direction}",

        (width - 230, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # TURN TIME
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Turn Time: {distraction_duration:.2f}s",

        (width - 230, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # DOWN TIME
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Down Time: {down_duration:.2f}s",

        (width - 230, 195),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # HEAD STATUS
    # --------------------------------------------------------

    if distraction_detected:

        head_status = "DISTRACTED"

        head_color = (0, 0, 255)


    elif down_detected:

        head_status = "POSSIBLE PHONE"

        head_color = (0, 165, 255)


    else:

        head_status = "NORMAL"

        head_color = (0, 255, 0)


    cv2.putText(

        frame,

        f"HEAD: {head_status}",

        (width - 270, 230),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        head_color,

        2

    )


    # ========================================================
    # MAIN RIDER STATUS
    # ========================================================

    if status == "ALERT":

        status_text = "STATUS: ALERT"

        status_color = (0, 255, 0)


    elif status == "EYES CLOSED":

        status_text = "STATUS: BLINK"

        status_color = (0, 255, 255)


    elif status == "DROWSY":

        status_text = "STATUS: DROWSY"

        status_color = (0, 0, 255)


    elif status == "DISTRACTED":

        status_text = "STATUS: DISTRACTED"

        status_color = (0, 0, 255)


    elif status == "POSSIBLE PHONE":

        status_text = "STATUS: POSSIBLE PHONE"

        status_color = (0, 165, 255)


    else:

        status_text = "STATUS: NO FACE"

        status_color = (0, 0, 255)


    # ========================================================
    # STATUS BACKGROUND
    # ========================================================

    cv2.rectangle(

        frame,

        (10, height - 60),

        (width - 10, height - 10),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        frame,

        status_text,

        (25, height - 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        status_color,

        2

    )


    # ========================================================
    # WINDOW
    # ========================================================

    cv2.imshow(

        "Stage 1 - Rider Monitoring",

        frame

    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Press Q to quit
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

if esp32 is not None and esp32.is_open:
    esp32.close()
    print("ESP32 serial connection closed.")

print()
print("========================================")
print("Stage 1 Rider Monitoring stopped.")
print(f"Total blinks : {blink_count}")
print(f"Total yawns  : {yawn_count}")
print("========================================")