# Kitne face samples har person ke liye capture karne hain
SAMPLES_PER_PERSON = 5

# Har sample ke darmiyan kitna wait karna hai (seconds mein)
CAPTURE_DELAY = 1.0

# Face samples kahan save honge (temporary storage)
DATASET_DIR = "dataset"

# SFace encoding model ka path
ENCODER_MODEL_PATH = "recognition/models/face_recognition_sface_2021dec.onnx"

# SQLite database ka path (registered logon ke naam + encodings yahan save hote hain)
DATABASE_PATH = "database/face_database.db"

# Recognition threshold (cosine similarity, SFace ke liye).
# Score is se ZYADA ho to match maana jayega, warna "Unknown".
# Is project ke dataset par: same insaan ~0.79-0.96, alag insaan ~0.32-0.44,
# isliye 0.55 beech mein safe jagah hai. Live camera par 'S' mode mein
# '+' / '-' keys se tune karo, phir yahan final value likh do.
RECOGNITION_THRESHOLD = 0.55

# Agar sab se achhe match aur doosre number ke match ka farq is se kam ho to
# "Unknown" dikhao (system confuse hai ke yeh A hai ya B). 0 = yeh check band.
MIN_MATCH_MARGIN = 0.05

# True = naam ke saath score bhi dikhao (jaise "Owais 0.82"). Threshold tune
# karte waqt True rakho, tune hone ke baad False kar sakte ho.
SHOW_SCORE = True