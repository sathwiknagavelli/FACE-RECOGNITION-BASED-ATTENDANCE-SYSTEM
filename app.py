import os
from flask import Flask, render_template, request, redirect, session, Response, flash, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from face_engine import generate_attendance
from database import connect_db, setup_database
import face_engine
from face_engine import process_video
from trainer import train_model
from flask import send_from_directory
app = Flask(__name__)
app.secret_key = "your_secret_key"

DATASET_DIR = "dataset"
UPLOADS_DIR = "uploads"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

setup_database()


# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))


# -------------------------
# REGISTER
# -------------------------
@app.route("/download_attendance")
def download_attendance():
    return send_from_directory(".", "attendance.csv", as_attachment=True)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        fullname = request.form["fullname"]
        username = request.form["username"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        conn = connect_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (email, fullname, username, password) VALUES (%s, %s, %s, %s)",
                (email, fullname, username, hashed_password)
            )
            conn.commit()
            flash("Registration successful. Please login.")
            return redirect(url_for("login"))
        except Exception as e:
            print("Register error:", e)
            flash("Username already exists.")
        finally:
            cursor.close()
            conn.close()

    return render_template("register.html")


# -------------------------
# LOGIN
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user[4], password):
            session["username"] = username
            flash("Login successful.")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.")

    return render_template("login.html")


# -------------------------
# DASHBOARD
# -------------------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


# -------------------------
# LOGOUT
# -------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------------
# ADD NEW PERSON
# -------------------------
@app.route("/add_person", methods=["POST"])
def add_person():
    if "username" not in session:
        return redirect(url_for("login"))

    name = request.form["name"].strip()
    images = request.files.getlist("images")

    if not name:
        flash("Person name is required.")
        return redirect(url_for("dashboard"))

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    saved_count = 0
    for img in images:
        if img and img.filename:
            filename = secure_filename(img.filename)
            img.save(os.path.join(person_dir, filename))
            saved_count += 1

    flash(f"{saved_count} images added for {name}.")
    return redirect(url_for("dashboard"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)
# -------------------------
# ADD IMAGES TO EXISTING PERSON
# -------------------------
@app.route("/add_images", methods=["POST"])
def add_images():
    if "username" not in session:
        return redirect(url_for("login"))

    name = request.form["name"].strip()
    images = request.files.getlist("images")

    person_dir = os.path.join(DATASET_DIR, name)

    if not os.path.exists(person_dir):
        flash("Person does not exist.")
        return redirect(url_for("dashboard"))

    saved_count = 0
    for img in images:
        if img and img.filename:
            filename = secure_filename(img.filename)
            img.save(os.path.join(person_dir, filename))
            saved_count += 1

    flash(f"{saved_count} more images added for {name}.")
    return redirect(url_for("dashboard"))


# -------------------------
# TRAIN MODEL
# -------------------------
@app.route("/train_model", methods=["POST"])
def train_face_model():
    if "username" not in session:
        return redirect(url_for("login"))

    success = train_model()

    if success:
        flash("Model trained successfully.")
    else:
        flash("Training failed. Please check dataset images.")

    return redirect(url_for("dashboard"))


# -------------------------
# UPLOAD VIDEO
# -------------------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "username" not in session:
        return redirect(url_for("login"))

    if "video" not in request.files:
        flash("No video file found.")
        return redirect(url_for("dashboard"))

    video = request.files["video"]

    if video.filename == "":
        flash("No file selected.")
        return redirect(url_for("dashboard"))

    filename = secure_filename(video.filename)
    path = os.path.join(UPLOADS_DIR, filename)
    video.save(path)

    return render_template("dashboard.html", video_file=filename)



@app.route("/recognized_names")
def recognized_names():
    if "username" not in session:
        return jsonify({"names": []})
    return jsonify({"names": face_engine.latest_recognized_names})


@app.route("/processing_status")
def processing_status():
    if "username" not in session:
        return jsonify({"status": "Not logged in"})
    return jsonify({"status": face_engine.latest_status})





# -------------------------
# RUN RECOGNITION
# -------------------------
@app.route("/run_recognition/<filename>")
def run_recognition(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    video_path = os.path.join(UPLOADS_DIR, filename)

    if not os.path.exists(video_path):
        return "Video not found.", 404

    return Response(
        process_video(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)