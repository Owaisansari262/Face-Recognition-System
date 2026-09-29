import cv2

def open_camera(camera_index=0):
    """
    Webcam ko open karta hai aur camera object return karta hai.
    camera_index=0 matlab default/built-in camera.
    """
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError("Camera open nahi ho saka. Check karo koi doosri app camera use nahi kar rahi.")

    return camera


def read_frame(camera):
    """
    Camera se ek frame (image) read karta hai.
    Return: (success: bool, frame: image)
    """
    success, frame = camera.read()
    return success, frame


def release_camera(camera):
    """
    Camera ko properly close karta hai. Yeh call karna zaroori hai
    warna camera 'busy' reh jata hai doosri apps ke liye.
    """
    camera.release()
