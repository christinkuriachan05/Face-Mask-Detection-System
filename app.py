from flask import Flask, render_template, Response

from live_camera import generate_frames


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# CAMERA STREAM
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("====================================")
    print("   FACE MASK DETECTION SYSTEM")
    print("====================================")
    print()
    print("Open this in your browser:")
    print()
    print("http://127.0.0.1:5000")
    print()
    print("====================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )