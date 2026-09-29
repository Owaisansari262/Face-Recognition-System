"""
Camera ke bina multiple logon par recognition test karta hai.

dataset/<naam>/sample_*.jpg mein jo images registration ke waqt save hui thin,
unhe use karke dekhta hai:
  - same insaan ke scores kitne aate hain
  - alag insaan ke scores kitne aate hain
  - kya do registered log aapas mein bohot milte julte hain
  - current threshold par kitni pehchan sahi/ghalat/Unknown hoti hai

Chalane ka tareeqa (project folder se):  python evaluate_dataset.py
"""
import os
import glob

import cv2
import numpy as np

import config
from recognition.encoder import load_encoder, get_encoding, average_encodings
from recognition.recognizer import cosine_similarity, recognize_face


def load_dataset_encodings(encoder):
    """Return: {naam: [encoding, encoding, ...]} — har sample ki alag encoding."""
    people = {}
    for name in sorted(os.listdir(config.DATASET_DIR)):
        folder = os.path.join(config.DATASET_DIR, name)
        if not os.path.isdir(folder):
            continue

        encodings = []
        for path in sorted(glob.glob(os.path.join(folder, "sample_*.jpg"))):
            image = cv2.imread(path)
            if image is not None:
                encodings.append(get_encoding(encoder, image))

        if encodings:
            people[name] = encodings
    return people


def build_known(people, test_name, test_index):
    """
    Test ke liye database banata hai. Jo sample test ho raha hai woh apne insaan ki
    average se bahar rakha jata hai (warna score jhoota barh jata hai).
    """
    known = {}
    for name, encodings in people.items():
        if name == test_name:
            others = [e for i, e in enumerate(encodings) if i != test_index]
            if others:
                known[name] = average_encodings(others)
        else:
            known[name] = average_encodings(encodings)
    return known


def main():
    if not os.path.isdir(config.DATASET_DIR):
        print("dataset/ folder nahi mila. Pehle main.py se logon ko register karo.")
        return

    encoder = load_encoder(config.ENCODER_MODEL_PATH)
    people = load_dataset_encodings(encoder)

    if len(people) < 2:
        print("Test ke liye kam az kam 2 registered log chahiye (dataset/ mein).")
        return

    names = list(people)
    print(f"\nLogon ki tadaad: {len(names)} -> {', '.join(names)}")
    print(f"Threshold: {config.RECOGNITION_THRESHOLD:.2f} | Min margin: {config.MIN_MATCH_MARGIN:.2f}\n")

    same_scores, diff_scores = [], []
    correct = unknown = wrong = 0
    matrix = {name: {other: [] for other in names} for name in names}

    for name, encodings in people.items():
        if len(encodings) < 2:
            print(f"({name} ke sirf {len(encodings)} sample hain, isay skip kiya)")
            continue

        for index, encoding in enumerate(encodings):
            known = build_known(people, name, index)

            for other, other_encoding in known.items():
                score = cosine_similarity(encoding, other_encoding)
                matrix[name][other].append(score)
                (same_scores if other == name else diff_scores).append(score)

            result, _, _ = recognize_face(
                encoding, known, config.RECOGNITION_THRESHOLD, config.MIN_MATCH_MARGIN
            )
            if result == name:
                correct += 1
            elif result == "Unknown":
                unknown += 1
            else:
                wrong += 1

    # ---- Score table: rows = asli insaan, columns = kis se compare hua ----
    width = max(len(n) for n in names) + 2
    print("Average score (row = asli insaan, column = kis registered insaan se compare hua)")
    print(" " * width + "".join(f"{n:>{width}}" for n in names))
    for name in names:
        row = "".join(
            f"{np.mean(matrix[name][other]):>{width}.2f}" if matrix[name][other] else f"{'-':>{width}}"
            for other in names
        )
        print(f"{name:<{width}}{row}")

    # ---- Summary ----
    print()
    if same_scores and diff_scores:
        same_min, diff_max = min(same_scores), max(diff_scores)
        print(f"Same insaan  : min {same_min:.2f}  mean {np.mean(same_scores):.2f}")
        print(f"Alag insaan  : max {diff_max:.2f}  mean {np.mean(diff_scores):.2f}")

        if same_min > diff_max:
            print(f"Suggested threshold: {(same_min + diff_max) / 2:.2f} (dono ke beech)")
        else:
            print("WARNING: same aur alag insaan ke scores overlap kar rahe hain.")
            print("Koi bhi threshold sab sahi nahi karega. Sab se pehle neeche wali duplicate warning dekho,")
            print("aur behtar roshni/seedha chehra ke saath dobara register karo.")

        total = correct + unknown + wrong
        print(f"\nCurrent threshold par ({total} tests):")
        print(f"  Sahi pehchan    : {correct}")
        print(f"  Unknown aaya    : {unknown}  (insaan registered tha, phir bhi Unknown)")
        print(f"  Ghalat naam     : {wrong}  (sab se buri halat)")

    # ---- Do logon ki encodings aapas mein bohot milti hain? ----
    averages = {name: average_encodings(encs) for name, encs in people.items()}
    warned = False
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            score = cosine_similarity(averages[first], averages[second])
            if score >= config.RECOGNITION_THRESHOLD:
                if not warned:
                    print("\nDUPLICATE WARNING:")
                    warned = True
                print(f"  {first} aur {second} ki encodings bohot milti hain ({score:.2f}).")
                print("  Shayad ek hi insaan do naam se register hai. Ek folder delete karo aur")
                print("  database se bhi hata do, ya sahi insaan ko dobara register karo.")

    print()


if __name__ == "__main__":
    main()