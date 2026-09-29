import numpy as np


def cosine_similarity(vector_a, vector_b):
    """
    Do vectors (encodings) kitne 'milte julte' hain — yeh number batata hai.
    Return: -1 se 1 ke darmiyan. 1 = bilkul same direction (same chehra),
    0 ke qareeb = koi taluq nahi.
    """
    a = vector_a.flatten()
    b = vector_b.flatten()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def find_best_match(encoding, known_encodings):
    """
    Ek encoding ko saare registered logon se compare karta hai.
    Return: (sab se qareeb naam, uska score, doosre number ke insaan ka score)
    Agar sirf ek hi banda registered ho to doosra score None hota hai.
    """
    scores = []
    for name, known_encoding in known_encodings.items():
        scores.append((cosine_similarity(encoding, known_encoding), name))

    scores.sort(reverse=True)  # sab se zyada score pehle

    best_score, best_name = scores[0]
    runner_up_score = scores[1][0] if len(scores) > 1 else None

    return best_name, best_score, runner_up_score


def recognize_face(encoding, known_encodings, threshold, min_margin=0.0):
    """
    Ek face ki encoding se decide karta hai ke woh kaun hai.

    Do jaanch hoti hain:
      1. Score threshold se kam ho to "Unknown" (koi bhi match kaafi qareeb nahi).
      2. Sab se achhe aur doosre number ke insaan ke score mein farq (margin)
         min_margin se kam ho to "Unknown" — matlab system confuse hai ke
         yeh A hai ya B, aur ghalat naam dene se behtar hai Unknown kehna.

    Return: (naam ya "Unknown", score, note)
    note: "ok", "low score", "ambiguous" ya "no data"
    """
    if not known_encodings:
        return "Unknown", 0.0, "no data"

    name, score, runner_up_score = find_best_match(encoding, known_encodings)

    if score < threshold:
        return "Unknown", score, "low score"

    if runner_up_score is not None and (score - runner_up_score) < min_margin:
        return "Unknown", score, "ambiguous"

    return name, score, "ok"