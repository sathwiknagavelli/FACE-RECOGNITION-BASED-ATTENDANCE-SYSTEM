import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

DATASET_DIR = "dataset"
MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(MODEL_DIR, exist_ok=True)


def is_image_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def get_embedding(image_path, model_name="Facenet"):
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend="retinaface",
            enforce_detection=False
        )
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error for {image_path}: {e}")
    return None


def train_model():
    X = []
    y = []

    if not os.path.exists(DATASET_DIR):
        print("Dataset folder not found.")
        return False

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            if not is_image_file(image_name):
                print(f"Skipping non-image file: {os.path.join(person_path, image_name)}")
                continue

            image_path = os.path.join(person_path, image_name)

            embedding = get_embedding(image_path)
            if embedding is not None:
                X.append(embedding)
                y.append(person_name)

    if len(X) < 2:
        print("Not enough valid training images found.")
        return False

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(X, y_encoded)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(classifier, f)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print("Model trained successfully.")
    print("Classes:", list(label_encoder.classes_))
    return True


if __name__ == "__main__":
    train_model()