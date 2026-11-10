import cv2, numpy as np
from collections import deque
from pathlib import Path
from tensorflow import keras

PROJ = Path(__file__).resolve().parent
_hits = list(PROJ.rglob("FERPlus_soft_96.keras"))
assert _hits, f"model not found under {PROJ}"
MODEL = _hits[0]
DETECT = PROJ/"models/face_detection_yunet_2023mar.onnx"

CLASSES = ["neutral","happiness","surprise","sadness","anger","disgust","fear"]
IMG_SIZE, SMOOTH, CONF = 96, 12, 0.7

print("model:", MODEL)
model = keras.models.load_model(MODEL)
prep  = keras.applications.efficientnet.preprocess_input
detector = cv2.FaceDetectorYN.create(str(DETECT), "", (320, 320), CONF)

history = deque(maxlen=SMOOTH)

def preprocess(face_gray):
    img = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype("float32")
    return prep(np.stack([img]*3, axis=-1))[np.newaxis, ...]

cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((W, H))
print(f"{W}x{H} — q quits, r resets smoothing")

try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, faces = detector.detect(frame)

        if faces is not None and len(faces):
            f = max(faces, key=lambda a: a[2]*a[3])
            x, y, w, h = [int(v) for v in f[:4]]
            x, y = max(x,0), max(y,0)
            crop = gray[y:y+h, x:x+w]

            if crop.size:
                probs = model.predict(preprocess(crop), verbose=0)[0]
                history.append(probs)
                avg = np.mean(history, axis=0)
                i = int(avg.argmax())

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{CLASSES[i]} {avg[i]:.2f}", (x, max(y-12,24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                for r, j in enumerate(avg.argsort()[::-1]):
                    yy = 30 + r*26
                    cv2.rectangle(frame, (10, yy-13), (10+int(avg[j]*190), yy+7),
                                  (60,140,220), -1)
                    cv2.putText(frame, f"{CLASSES[j]:<10} {avg[j]:.2f}", (14, yy+3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        else:
            history.clear()
            cv2.putText(frame, "no face", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("emotion — q quits, r resets", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("r"): history.clear()
finally:
    cap.release(); cv2.destroyAllWindows()