import cv2
import numpy as np
import os
import time
import threading

# Windows sound
try:
    import winsound
except ImportError:
    winsound = None

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mask_detector.keras"
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "face_detection",
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# LOAD MASK MODEL
# ============================================================

print("Loading mask detection model...")

model = load_model(MODEL_PATH)

print("Mask model loaded successfully.")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print("Loading face detector...")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise FileNotFoundError(
        "Haar Cascade could not be loaded:\n"
        + CASCADE_PATH
    )

print("Face detector loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not be opened."
    )

print("Camera opened successfully.")


# ============================================================
# BEEP CONTROL
# ============================================================

last_beep_time = 0


def beep_beep():

    global last_beep_time

    current_time = time.time()

    # Beep every 2 seconds
    if current_time - last_beep_time < 2:
        return

    last_beep_time = current_time

    if winsound is None:
        return

    def make_beep():

        try:

            winsound.Beep(1000, 200)

            time.sleep(0.15)

            winsound.Beep(1000, 200)

        except Exception:
            pass

    threading.Thread(
        target=make_beep,
        daemon=True
    ).start()


# ============================================================
# GENERATE CAMERA FRAMES
# ============================================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera.")

            break


        # ----------------------------------------------------
        # FLIP CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # DETECT FACES
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ====================================================
        # PROCESS EACH FACE
        # ====================================================

        for (x, y, w, h) in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]


            if face.size == 0:
                continue


            # ------------------------------------------------
            # RESIZE FACE
            # ------------------------------------------------

            face = cv2.resize(
                face,
                (128, 128)
            )


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            face = face.astype(
                "float32"
            ) / 255.0


            # ------------------------------------------------
            # ADD BATCH DIMENSION
            # ------------------------------------------------

            face = np.expand_dims(
                face,
                axis=0
            )


            # =================================================
            # MODEL PREDICTION
            # =================================================

            prediction = model.predict(
                face,
                verbose=0
            )


            # =================================================
            # DETERMINE MASK / NO MASK
            # =================================================

            if prediction.shape[-1] == 2:

                mask_probability = float(
                    prediction[0][0]
                )

                no_mask_probability = float(
                    prediction[0][1]
                )


                if mask_probability > no_mask_probability:

                    # =========================================
                    # MASK DETECTED
                    # =========================================

                    label = "MASK DETECTED"

                    confidence = mask_probability

                    box_color = (
                        0,
                        255,
                        0
                    )


                    # Green message

                    cv2.putText(
                        frame,
                        "MASK DETECTED",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )


                else:

                    # =========================================
                    # NO MASK
                    # =========================================

                    label = "WEAR MASK"

                    confidence = no_mask_probability

                    box_color = (
                        0,
                        0,
                        255
                    )


                    # Beep

                    beep_beep()


                    # Warning message

                    cv2.putText(
                        frame,
                        "PLEASE WEAR A MASK",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )


            else:

                # =================================================
                # BINARY MODEL
                # =================================================

                probability = float(
                    prediction[0][0]
                )


                if probability < 0.5:

                    # =============================================
                    # MASK DETECTED
                    # =============================================

                    label = "MASK DETECTED"

                    confidence = 1 - probability

                    box_color = (
                        0,
                        255,
                        0
                    )


                    cv2.putText(
                        frame,
                        "MASK DETECTED",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )


                else:

                    # =============================================
                    # NO MASK
                    # =============================================

                    label = "WEAR MASK"

                    confidence = probability

                    box_color = (
                        0,
                        0,
                        255
                    )


                    # Beep

                    beep_beep()


                    # Warning message

                    cv2.putText(
                        frame,
                        "PLEASE WEAR A MASK",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )


            # =================================================
            # DRAW FACE RECTANGLE
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                3
            )


            # =================================================
            # LABEL ABOVE FACE
            # =================================================

            text = "{} {:.1f}%".format(
                label,
                confidence * 100
            )


            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                box_color,
                2
            )


        # ====================================================
        # CAMERA TITLE
        # ====================================================

        cv2.putText(
            frame,
            "LIVE FACE MASK DETECTION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # ENCODE FRAME
        # ====================================================

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not success:
            continue


        frame_bytes = buffer.tobytes()


        # ====================================================
        # SEND TO BROWSER
        # ====================================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# RELEASE CAMERA
# ============================================================

def release_camera():

    global camera

    if camera is not None:

        camera.release()

        camera = None

    cv2.destroyAllWindows()

    print("Camera stopped.")