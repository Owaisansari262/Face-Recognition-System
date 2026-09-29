import cv2

# Haar Cascade model OpenCV ke saath pehle se aata hai — humein khud train nahi karna
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_faces(frame):
    """
    Frame mein chehre dhoondta hai.
    Return: list of boxes, har box = (x, y, width, height)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # image ko kitne steps mein chota kar ke check karna hai
        minNeighbors=5,       # zyada value = kam false detections, lekin real face bhi miss ho sakta hai
        minSize=(60, 60)      # itne se chote face ko ignore karo (noise)
    )

    return faces


def draw_faces(frame, faces, label="Face"):
    """
    Har detected face ke around box aur uske upar naam draw karta hai.
    label: ya to ek string (sab faces ke liye same), ya list of strings
           (har face ke liye alag naam, faces ke order mein).
    "Unknown" laal rang mein dikhta hai, baaqi sab hare mein.
    """
    for index, (x, y, w, h) in enumerate(faces):
        text = label[index] if isinstance(label, (list, tuple)) else label
        color = (0, 0, 255) if text.startswith("Unknown") else (0, 255, 0)  # BGR

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Text ka size nikalo taake peeche background box bana sakein
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        # Naam face ke upar; agar face frame ke top par hai to box ke andar neeche
        if y - text_h - baseline - 6 >= 0:
            top = y - text_h - baseline - 6
        else:
            top = y + h
        bottom = top + text_h + baseline + 6

        cv2.rectangle(frame, (x, top), (x + text_w + 8, bottom), color, -1)
        cv2.putText(
            frame, text, (x + 4, bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

    return frame


def crop_face(frame, box):
    """
    Frame se sirf face wala hissa cut (crop) karta hai.
    box = (x, y, width, height) jo detect_faces se aata hai.
    """
    x, y, w, h = box
    return frame[y:y + h, x:x + w]