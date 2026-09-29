# Face Recognition System (Python)

Beginner-friendly, camera-based face recognition. Pehle kisi ka naam aur chehra register karo,
phir jab woh camera ke samne aaye to system uska naam chehre ke upar dikhata hai.
Koi match na mile to **Unknown** dikhata hai.

Yeh ek local learning project hai: face data sirf aap ke computer par rehta hai, kisi service par upload nahi hota.

## Technology

- **Python** - main language
- **OpenCV** - webcam, face detection (Haar Cascade), drawing, face encoding model (SFace)
- **NumPy** - vectors aur similarity ka hisaab
- **SQLite** - registered logon ke naam aur encodings (raw webcam images database mein nahi jati)

## Installation (Windows)

Project folder mein PowerShell kholo:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Agar `activate` par "running scripts is disabled" aaye:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```

Activate hone ke baad prompt ke shuru mein `(venv)` dikhna chahiye. Har nayi terminal window mein
pehle `venv\Scripts\activate` karna hota hai.

Face encoder model `recognition/models/face_recognition_sface_2021dec.onnx` mein maujood hona chahiye.

## Chalane ka tareeqa

```powershell
python main.py
```

| Key | Kaam |
|---|---|
| `R` | Naya person register karo (terminal mein naam likho, camera ke samne aao, **SPACE** dabao) |
| `S` | Recognition ON / OFF |
| `+` / `-` | Recognition threshold live barhao / ghatao (sirf chalte program ke liye) |
| `Q` | Quit |

### Registration
1. `R` dabao, terminal mein naam likho, Enter.
2. Camera window mein sirf **wohi insaan** samne aaye. Ready ho to **SPACE** dabao.
3. 5 samples automatically capture hote hain (frame mein sirf ek chehra hona chahiye).
4. Wohi naam dobara register karoge to purani encoding update ho jayegi.

### Recognition
`S` dabao. Har chehre ke upar naam aur score dikhta hai, jaise `Owais 0.93` (hara) ya `Unknown 0.28` (laal).
Ek se zyada log hon to har chehra alag pehchana jata hai.

## Project structure

```text
face-recognition-system/
|-- main.py                 program ka entry point, keys aur main loop
|-- config.py               saari settings (threshold, samples, paths)
|-- evaluate_dataset.py     camera ke bina multi-person test
|-- requirements.txt
|-- README.md
|-- database/
|   |-- db.py               SQLite: table banana, save, load
|   `-- face_database.db    (pehli baar chalane par khud ban jati hai)
|-- recognition/
|   |-- detector.py         face detection, box aur naam draw karna, face crop
|   |-- encoder.py          face -> 128 numbers (embedding)
|   |-- recognizer.py       similarity aur Unknown ka faisla
|   `-- models/             SFace ONNX model
|-- utils/
|   `-- camera.py           webcam open, frame read, release
`-- dataset/                registration ke samples (naam ke hisaab se folders)
```

## Yeh kaise kaam karta hai

```text
Webcam frame -> face detect -> face crop -> encoding (128 numbers)
             -> registered logon se cosine similarity -> naam ya Unknown -> window par draw
```

- **Frame:** camera ki har image. Video dar asal bohot saare frames ka silsila hai.
- **Embedding:** chehre ka "fingerprint", 128 numbers ka vector. Ek hi insaan ke vectors qareeb hote hain.
- **Cosine similarity:** do vectors kitne milte hain, -1 se 1 tak. 1 = bilkul same.
- **Threshold:** is score se kam ho to Unknown.
- **Margin:** agar sab se achhe aur doosre number ke insaan mein farq bohot kam ho to bhi Unknown
  (system confuse hai, ghalat naam dene se behtar hai).

## Threshold tune karna

`config.py` mein:

| Setting | Matlab |
|---|---|
| `RECOGNITION_THRESHOLD` | is se kam score = Unknown (SFace ke liye 0.363 recommended start, is project mein 0.55) |
| `MIN_MATCH_MARGIN` | top-2 scores ka kam az kam farq (0 = check band) |
| `SHOW_SCORE` | naam ke saath score dikhao ya nahi |

Kaise tune karein:
1. `python main.py`, phir `S`.
2. Sirf khud ko dikhao aur different angles/roshni mein scores note karo (sab se kam score yaad rakho).
3. Kisi doosre ko dikhao aur sab se zyada score note karo.
4. Threshold in dono ke beech rakho. Ghalat pehchan se bachna zyada zaroori ho to upar ki taraf.
5. `+` / `-` se live check karo, phir final value `config.py` mein likho.

Threshold barhane se ghalat pehchan kam hoti hai lekin "Unknown" (registered insaan ko na pehchanna) barh sakta hai.
Har model/library ke liye threshold alag hota hai, isliye kisi aur project ki value seedha copy mat karo.

## Multiple people test

Jab 2 ya zyada log register ho jayein:

```powershell
python evaluate_dataset.py
```

Yeh `dataset/` ke samples se dikhata hai: same/alag insaan ke score, suggested threshold,
current threshold par sahi/Unknown/ghalat kitne hain, aur agar do logon ki encodings bohot milti hon
(jaise ek hi insaan do naam se register ho) to DUPLICATE WARNING.

## Masle aur hal

| Masla | Hal |
|---|---|
| `No module named 'cv2'` | venv activate nahi hai ya packages install nahi hue. `venv\Scripts\activate`, phir `pip install -r requirements.txt` |
| Camera nahi khulta | Koi doosri app (Zoom, Teams) camera use kar rahi ho to band karo. Agar laptop mein ek se zyada camera hain to `utils/camera.py` mein `camera_index=0` ko `1` karke dekho |
| Sab Unknown aa raha hai | Roshni theek karo, seedha camera ki taraf dekho, threshold `-` se thora kam karo, ya dobara register karo |
| Kisi doosre ko pehchan leta hai | Threshold `+` se barhao, ya zyada roshni/seedhe chehre ke saath dobara register karo |
| Do logon mein confusion | `python evaluate_dataset.py` chalao, DUPLICATE WARNING dekho |
| Chehra detect nahi hota | Chehra frame mein poora aur roshan ho, camera se bohot door na ho |

## Security aur privacy

- Face encodings local SQLite mein rehti hain, kahin upload nahi hoti.
- Raw webcam images database mein save nahi hoti. `dataset/` mein registration ke crop kiye hue face samples rehte hain;
  agar nahi chahiye to registration ke baad woh folders delete kar sakte ho (recognition database se chalti hai).
- Webcam ya database kisi API ke through expose nahi hai.
- `database/face_database.db` aur `dataset/` ko kisi ke saath share ya git par push mat karo.

## Future version: web application (abhi implement nahi kiya)

Jab desktop version achhi tarah chal jaye, isay web app mein yun convert kar sakte hain:

```text
React (browser: camera, UI)
   |  HTTP / WebSocket
FastAPI (Python API server)
   |
Python Face Recognition (yehi detector / encoder / recognizer)
   |
SQLite / PostgreSQL
```

- **React:** browser mein webcam kholta hai (`getUserMedia`), frames ya images API ko bhejta hai, aur jawab mein naam aur box dikhata hai.
- **FastAPI:** endpoints jaise `POST /register` (naam + images) aur `POST /recognize` (image -> naam, score, box).
  `recognition/` aur `database/` ka code kaafi had tak wahi rehta hai, sirf `main.py` ki jagah API layer aati hai.
- **Database:** SQLite se PostgreSQL par jaana ho to `database/db.py` badalna hoga.
- **Dhyan dene wali baatein:** face data sensitive hai. Web par HTTPS, login/authentication, kis ko data dekhne ka haq hai,
  aur consent zaroori hain. Camera ya database ko bina authentication ke expose mat karna.