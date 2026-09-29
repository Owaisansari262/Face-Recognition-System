import cv2
import os
import time
from utils.camera import open_camera, read_frame, release_camera
from recognition.detector import detect_faces, draw_faces, crop_face
from recognition.encoder import load_encoder, get_encoding, average_encodings
from database.db import init_db, save_person, load_all_persons
from recognition.recognizer import recognize_face
import config


def wait_for_ready(camera, name):
    """
    Live preview dikhata hai jab tak SPACE na dabaya jaye.
    Is dauran koi sample capture nahi hota — isse dost ko camera ke samne
    aane ka time mil jata hai.
    Return: True agar SPACE dabaya, False agar Q se cancel kiya.
    """
    print(f"\n{name} ko camera ke samne bithao.")
    print("Jab ready ho jao to camera window par SPACE dabao. (Cancel: Q)\n")

    while True:
        success, frame = read_frame(camera)
        if not success:
            print("Frame read nahi hua.")
            return False

        faces = detect_faces(frame)
        display_frame = draw_faces(frame.copy(), faces, label=name)
        cv2.putText(
            display_frame, "SPACE = capture shuru | Q = cancel", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.imshow("Face Recognition System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        elif key == ord('q'):
            return False


def register_person(camera, encoder):
    name = input("Enter person's name: ").strip()
    if not name:
        print("Naam khali nahi ho sakta. Registration cancel.")
        return

    # Pehle ready ka wait karo — capture tab tak shuru nahi hoga
    if not wait_for_ready(camera, name):
        print("Registration cancel ho gaya.")
        return

    save_dir = os.path.join(config.DATASET_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    # Is naam ke purane samples hata do taake naye/purane images mix na hon
    for old_file in os.listdir(save_dir):
        if old_file.startswith("sample_") and old_file.endswith(".jpg"):
            os.remove(os.path.join(save_dir, old_file))

    print("Face samples automatically capture honge...\n")

    samples_captured = 0
    last_capture_time = time.time()
    captured_encodings = []

    while samples_captured < config.SAMPLES_PER_PERSON:
        success, frame = read_frame(camera)
        if not success:
            print("Frame read nahi hua.")
            break

        faces = detect_faces(frame)
        display_frame = draw_faces(frame.copy(), faces, label="Capturing...")
        cv2.imshow("Face Recognition System", display_frame)

        if len(faces) == 1:
            current_time = time.time()
            if current_time - last_capture_time >= config.CAPTURE_DELAY:
                face_crop = crop_face(frame, faces[0])

                sample_path = os.path.join(save_dir, f"sample_{samples_captured + 1}.jpg")
                cv2.imwrite(sample_path, face_crop)

                # Isi sample se encoding bhi generate kar lo
                encoding = get_encoding(encoder, face_crop)
                captured_encodings.append(encoding)

                samples_captured += 1
                last_capture_time = current_time
                print(f"Sample {samples_captured}/{config.SAMPLES_PER_PERSON} capture ho gaya.")

        elif len(faces) > 1:
            print("Sirf ek chehra frame mein hona chahiye registration ke waqt.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Registration cancel ho gaya.")
            return

    if not captured_encodings:
        print("Koi sample capture nahi hua. Registration cancel.")
        return

    # Sab samples ke encodings ka average nikaal ke save karo
    final_encoding = average_encodings(captured_encodings)
    save_person(config.DATABASE_PATH, name, final_encoding)

    print(f"\nRegistration successful! {name} ke {samples_captured} samples save ho gaye.")
    print(f"{name} ka face encoding database mein save ho gaya.\n")


def show_registered_people(known_encodings):
    """Terminal mein dikhata hai ke abhi kaun kaun registered hai."""
    if not known_encodings:
        print("Abhi koi person registered nahi hai. 'R' dabao aur register karo.")
        return

    print(f"Registered people ({len(known_encodings)}):")
    for name, encoding in known_encodings.items():
        print(f"  - {name} (encoding size: {len(encoding)})")


def main():
    print("Camera open ho rahi hai...")
    camera = open_camera()

    print("Face encoder model load ho raha hai...")
    encoder = load_encoder(config.ENCODER_MODEL_PATH)

    # Database taiyar karo (pehli baar chalane par table ban jati hai)
    init_db(config.DATABASE_PATH)

    # Saved encodings ek baar memory mein load kar lo (baar baar file na parhni pare)
    known_encodings = load_all_persons(config.DATABASE_PATH)
    show_registered_people(known_encodings)

    print("Camera window khul gayi.")
    print("Controls: 'R' = Register | 'S' = Recognition on/off | '+' / '-' = Threshold | 'Q' = Quit")
    print(f"Threshold: {config.RECOGNITION_THRESHOLD:.2f} | Min margin: {config.MIN_MATCH_MARGIN:.2f}\n")

    # Threshold live badalne ke liye alag variable (config.py wali value se shuru hota hai)
    threshold = config.RECOGNITION_THRESHOLD
    recognition_on = False
    last_print_time = 0

    while True:
        success, frame = read_frame(camera)
        if not success:
            print("Frame read nahi hua, camera check karo.")
            break

        faces = detect_faces(frame)
        # Recognition mode: har face ko alag alag pehchano
        labels = "Face"
        if recognition_on:
            results = []
            labels = []
            for face_box in faces:
                face_crop = crop_face(frame, face_box)
                encoding = get_encoding(encoder, face_crop)
                name, score, note = recognize_face(
                    encoding, known_encodings, threshold, config.MIN_MATCH_MARGIN
                )
                labels.append(f"{name} {score:.2f}" if config.SHOW_SCORE else name)
                if note in ("ok", "no data"):
                    results.append(f"{name} ({score:.2f})")
                else:
                    results.append(f"{name} ({score:.2f}, {note})")

            # Terminal spam na ho, isliye har 1 second mein sirf ek baar print karo
            if results and time.time() - last_print_time >= 1.0:
                print("Recognized:", " | ".join(results))
                last_print_time = time.time()

        frame = draw_faces(frame, faces, label=labels)
        cv2.imshow("Face Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            register_person(camera, encoder)
            # Naya person register hua hai, list dobara load karo
            known_encodings = load_all_persons(config.DATABASE_PATH)
            show_registered_people(known_encodings)
        elif key == ord('s'):
            recognition_on = not recognition_on
            if recognition_on and not known_encodings:
                print("Koi person registered nahi hai, sab 'Unknown' aayenge. Pehle 'R' se register karo.")
            print("Recognition ON" if recognition_on else "Recognition OFF")
        elif key in (ord('+'), ord('=')):
            threshold = min(1.0, threshold + 0.02)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.0, threshold - 0.02)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('q'):
            print("Quit ho rahe hain...")
            break

    release_camera(camera)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()