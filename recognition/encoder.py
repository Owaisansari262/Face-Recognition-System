import cv2
import numpy as np
import os


def load_encoder(model_path):
    """
    SFace encoding model load karta hai.
    Return: encoder object jo feature() method deta hai.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file nahi mili: {model_path}\n"
            "Pehle SFace .onnx model download karo aur recognition/models/ mein rakho."
        )

    encoder = cv2.FaceRecognizerSF.create(
        model=model_path,
        config="",
        backend_id=0,
        target_id=0
    )
    return encoder


def get_encoding(encoder, face_image):
    """
    Ek face image se 128-number ka encoding (vector) banata hai.
    face_image: cropped face (BGR image)
    Return: numpy array (encoding vector)
    """
    resized_face = cv2.resize(face_image, (112, 112))
    encoding = encoder.feature(resized_face)
    return encoding


def average_encodings(encodings_list):
    """
    Multiple samples ke encodings ka average nikalta hai — isse
    ek zyada 'stable' aur reliable encoding milta hai us person ke liye.
    """
    stacked = np.vstack(encodings_list)
    return np.mean(stacked, axis=0)