import cv2
import numpy as np
import time
import winsound
import pyttsx3
import threading

from tensorflow.keras.models import load_model


# =========================================================
# FILE PATHS
# =========================================================

FACE_CASCADE_PATH = (
    "face_detection/haarcascade_frontalface_default.xml"
)

MODEL_PATH = "model/mask_detector.keras"


# =========================================================
# LOAD FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)

if face_detector.empty():
    raise RuntimeError(
        "Haar Cascade could not be loaded.\n"
        "Check:\n"
        + FACE_CASCADE_PATH
    )

print("Face detector loaded.")


# =========================================================
# LOAD MASK MODEL
# =========================================================

model = load_model(MODEL_PATH)

print("Mask model loaded.")


# =========================================================
# VOICE
# =========================================================

try:

    speaker = pyttsx3.init()

    speaker.setProperty("rate", 150)
    speaker.setProperty("volume", 1.0)

    voice_available = True

    print("Voice system loaded.")

except Exception as e:

    print("Voice unavailable:", e)

    speaker = None

    voice_available = False


# =========================================================
# WARNING CONTROL
# =========================================================

last_warning_time = 0

WARNING_INTERVAL = 5

warning_lock = threading.Lock()


def warning_sound():

    global last_warning_time

    current_time = time.time()

    if (
        current_time - last_warning_time
        < WARNING_INTERVAL
    ):
        return

    last_warning_time = current_time

    def play_warning():

        try:

            # Beep beep
            winsound.Beep(1000, 300)
            winsound.Beep(1000, 300)

            # Voice
            if voice_available:

                speaker.say(
                    "Please wear a mask"
                )

                speaker.runAndWait()

        except Exception as e:

            print("Warning error:", e)

    # Run warning separately
    threading.Thread(
        target=play_warning,
        daemon=True
    ).start()


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    raise RuntimeError(
        "Camera could not be opened."
    )

print("Camera opened.")


# =========================================================
# CAMERA FRAME GENERATOR
# =========================================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read camera frame."
            )

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # =====================================================
        # FACE DETECTION
        # =====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # =====================================================
        # PROCESS FACES
        # =====================================================

        for (x, y, w, h) in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:
                continue


            # Convert BGR → RGB
            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )


            # Resize
            face_resized = cv2.resize(
                face_rgb,
                (128, 128)
            )


            # Normalize
            face_array = (
                face_resized.astype(
                    "float32"
                ) / 255.0
            )


            # Add batch dimension
            face_array = np.expand_dims(
                face_array,
                axis=0
            )


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                face_array,
                verbose=0
            )


            confidence = float(
                prediction[0][0]
            )


            # =================================================
            # MASK RESULT
            # =================================================

            if confidence >= 0.5:

                label = "WITHOUT MASK"

                color = (
                    0,
                    0,
                    255
                )

                display_confidence = (
                    confidence
                )

                # Warning
                warning_sound()

            else:

                label = "MASK"

                color = (
                    0,
                    255,
                    0
                )

                display_confidence = (
                    1 - confidence
                )


            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )


            # =================================================
            # LABEL
            # =================================================

            text = (
                f"{label} "
                f"{display_confidence * 100:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


        # =====================================================
        # WARNING ON VIDEO
        # =====================================================

        without_mask_found = False

        for (x, y, w, h) in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face_resized = cv2.resize(
                face_rgb,
                (128, 128)
            )

            face_array = (
                face_resized.astype(
                    "float32"
                ) / 255.0
            )

            face_array = np.expand_dims(
                face_array,
                axis=0
            )

            prediction = model.predict(
                face_array,
                verbose=0
            )

            confidence = float(
                prediction[0][0]
            )

            if confidence >= 0.5:

                without_mask_found = True

                break


        if without_mask_found:

            # Red warning bar
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 80),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "PLEASE WEAR A MASK!",
                (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )


        # =====================================================
        # FACE COUNT
        # =====================================================

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =====================================================
        # ENCODE JPEG
        # =====================================================

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue


        frame_bytes = buffer.tobytes()


        # =====================================================
        # SEND TO FLASK
        # =====================================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================================
# RELEASE CAMERA
# =========================================================

def release_camera():

    if camera is not None:

        camera.release()

    cv2.destroyAllWindows()

    print("Camera released.")