import os
import cv2
import pickle
import numpy as np
import csv
from deepface import DeepFace

MODEL_DIR = "models"
DATASET_DIR = "dataset"

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

latest_recognized_names = []
latest_status = "Idle"

recognized_set = set()


def reset_recognition():
    global latest_recognized_names, latest_status, recognized_set
    latest_recognized_names = []
    latest_status = "Idle"
    recognized_set.clear()


def generate_attendance():
    students = os.listdir(DATASET_DIR)

    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Roll No", "Status"])

        for student in students:
            status = "Present" if student in recognized_set else "Absent"
            writer.writerow([student, status])


def load_models():
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None, None

    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return classifier, label_encoder


def get_face_embedding(face_img):
    try:
        embedding_objs = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False
        )
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print("Embedding error:", e)
    return None


def recognize_faces(frame, classifier, label_encoder, confidence_threshold=0.75):
    results_list = []

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print("Detection error:", e)
        return results_list

    for face_obj in detections:
        facial_area = face_obj.get("facial_area", {})
        x = int(facial_area.get("x", 0))
        y = int(facial_area.get("y", 0))
        w = int(facial_area.get("w", 0))
        h = int(facial_area.get("h", 0))

        if w <= 0 or h <= 0:
            continue

        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        embedding = get_face_embedding(face_img)
        if embedding is None:
            continue

        probs = classifier.predict_proba([embedding])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        name = label_encoder.inverse_transform([pred_idx])[0]

        if confidence < confidence_threshold:
            name = "Unknown"

        results_list.append((x, y, w, h, name, confidence))

    return results_list


def process_video(video_path):
    global latest_recognized_names, latest_status, recognized_set

    classifier, label_encoder = load_models()

    if classifier is None or label_encoder is None:
        latest_status = "Model not found. Train first."
        return

    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    skip_frames = 12
    output_width = 854
    output_height = 480
    cached_results = []

    latest_recognized_names = []
    recognized_set.clear()
    latest_status = "Processing started"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        if height > width:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.resize(frame, (output_width, output_height))
        frame_index += 1

        if frame_index % skip_frames == 0 or not cached_results:
            latest_status = "Detecting faces"
            cached_results = recognize_faces(frame, classifier, label_encoder)

            names = []
            for (_, _, _, _, name, _) in cached_results:
                if name != "Unknown":
                    if name not in recognized_set:
                        recognized_set.add(name)
                    if name not in names:
                        names.append(name)

            latest_recognized_names = names
            latest_status = "Streaming recognition"

        for (x, y, w, h, name, confidence) in cached_results:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()
    latest_status = "Completed"

    generate_attendance()