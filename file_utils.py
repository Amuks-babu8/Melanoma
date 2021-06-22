import pickle

from tensorflow.python.keras.models import load_model

import constant_values


def save_labels(label):
    print("[INFO] saving labels...")
    pickle.dump(label, open(constant_values.LABELS_FILE, "wb"))
    print("[INFO] Saving labels complete")


def save_model(model):
    print("[INFO] Saving model...")
    model.save(constant_values.MODEL_FILE)
    print("[INFO] Model Saved ")


def load_labels():
    print("[INFO] Loading labels...")
    labels = pickle.load(open(constant_values.LABELS_FILE, "rb"))
    print("[INFO] Loading labels complete")
    return labels


def load_saved_model():
    print("[INFO] Loading saved model... ")
    model = load_model(constant_values.MODEL_FILE)
    print("[INFO] Loading Model Complete")
    return model
