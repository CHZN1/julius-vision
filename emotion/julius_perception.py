"""Vision thread: identity + emotion, gated for conversational use."""
import cv2, numpy as np, torch, torch.nn as nn, threading, time
from collections import deque
from pathlib import Path
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis

ART = Path(__file__).resolve().parent/"artifacts"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

CLASSES = ["surprise","fear","disgust","happiness","sadness","anger","neutral"]
SMOOTH, DETECT_EVERY, ID_THRESH, TARGET_FPS = 15, 4, 0.40, 8

NUDGE_EMO = {"anger","sadness","fear"}
HOLD_SEC  = {"anger":60, "fear":60, "sadness":110}   # sadness longer: resting face reads sad
CONF_MIN  = {"anger":0.55, "fear":0.60, "sadness":0.65}
COOLDOWN  = 20*60


class FERNet(nn.Module):
    def __init__(s, n=7):
        super().__init__()
        s.backbone = InceptionResnetV1(pretrained=None)
        s.head = nn.Sequential(nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,n))
    def forward(s, x): return s.head(s.backbone(x))


class Perception:
    def __init__(s, show_window=False, target_fps=TARGET_FPS, emotion_interval=0.0, camera_source=0, allowed_modules=None):
        s.target_fps = max(1, float(target_fps))
        s.emotion_interval = max(0.0, float(emotion_interval))
        rate = min(s.target_fps, 1/s.emotion_interval) if s.emotion_interval else s.target_fps
        s.smooth_count = max(2, round(SMOOTH / TARGET_FPS * rate))
        s.camera_source = camera_source
        s.camera_available = False
        s._thread = None
        s.show = show_window            # console status line, not a GUI window
        s.state = {"person":None, "id_conf":0.0, "emotion":None, "conf":0.0,
                   "held":0.0, "present":0.0, "seen":False}
        s.lock = threading.Lock()
        s._stop = threading.Event()
        s._cur, s._since = None, 0.0
        s._last_nudge, s._last_kind = -1e9, None
        s._first_seen = None

        def _load(name):
            m = FERNet().to(DEV)
            m.load_state_dict(torch.load(ART/f"models/{name}.pt", map_location=DEV),
                              strict=False)
            return m.eval()

        s.m1, s.m2 = _load("VGGFace2_ft"), _load("AffNet_RAFDB")
        s.app = FaceAnalysis(name="buffalo_l", allowed_modules=allowed_modules,
                             providers=["CoreMLExecutionProvider","CPUExecutionProvider"])
        s.app.prepare(ctx_id=-1, det_size=(320,320))
        db = np.load(ART/"face_db.npz", allow_pickle=True)
        s.db_vecs, s.db_names = db["vecs"], db["names"]
        s.tf = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor(),
                                   transforms.Normalize([0.5]*3, [0.5]*3)])

    @torch.no_grad()
    def _emotion(s, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = s.tf(Image.fromarray(rgb)).unsqueeze(0).to(DEV)
        p1 = torch.softmax(s.m1(x), 1).cpu().numpy()[0]
        p2 = torch.softmax(s.m2(x), 1).cpu().numpy()[0]
        g = np.exp((np.log(np.clip(p1,1e-9,1)) + np.log(np.clip(p2,1e-9,1)))/2)
        return g/g.sum()

    def _identify(s, vec):
        sims = s.db_vecs @ vec
        i = int(sims.argmax())
        return (str(s.db_names[i]) if sims[i] >= ID_THRESH else "unknown"), float(sims[i])

    def _loop(s):
        cap = s._cap
        hist, bbox, ident, n = deque(maxlen=s.smooth_count), None, (None, 0.0), 0
        delay = 1.0/s.target_fps
        last_emotion = -1e9
        while not s._stop.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if isinstance(s.camera_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                s._stop.wait(0.1); continue
            n += 1

            if n % DETECT_EVERY == 1:
                faces = s.app.get(frame)
                if faces:
                    f = max(faces, key=lambda a:(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1]))
                    bbox = tuple(max(int(v),0) for v in f.bbox)
                    ident = s._identify(f.normed_embedding)
                    if s._first_seen is None: s._first_seen = time.time()
                else:
                    bbox = None; hist.clear(); s._first_seen = None

            now = time.time()
            if bbox:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size and now - last_emotion >= s.emotion_interval:
                    hist.append(s._emotion(crop))
                    last_emotion = now
                if hist:
                    avg = np.mean(hist, axis=0)
                    emo, conf = CLASSES[int(avg.argmax())], float(avg.max())
                    if emo != s._cur:
                        s._cur, s._since = emo, now
                    with s.lock:
                        s.state.update(person=ident[0], id_conf=ident[1],
                                       emotion=emo, conf=conf,
                                       held=now - s._since,
                                       present=now - (s._first_seen or now),
                                       seen=True)
            else:
                with s.lock:
                    s.state.update(seen=False, emotion=None, conf=0.0)

            # console status only — cv2.imshow crashes off the main thread on macOS
            if s.show and n % 30 == 0:
                st = s.snapshot()
                if st["seen"]:
                    print(f"\r  [{st['person']} · {st['emotion']} {st['conf']:.2f}"
                          f" · {st['held']:.0f}s]      ", end="", flush=True)

            time.sleep(max(0, delay - (time.time()-t0)))
        cap.release()

    def start(s):
        # macOS camera authorization must be requested on the main thread.
        s._cap = cv2.VideoCapture(s.camera_source)
        s.camera_available = s._cap.isOpened()
        if not s.camera_available:
            s._cap.release()
            print("  [camera unavailable — vision inactive]", flush=True)
            return s
        s._thread = threading.Thread(target=s._loop, daemon=True)
        s._thread.start()
        return s

    def stop(s):
        s._stop.set()
        if s._thread is not None:
            s._thread.join(timeout=5)

    def snapshot(s):
        with s.lock: return dict(s.state)

    def check_nudge(s):
        """Returns an emotion name if a wellbeing check-in is warranted, else None."""
        st = s.snapshot(); now = time.time()
        e, c, h = st["emotion"], st["conf"], st["held"]
        if not st["seen"] or e not in NUDGE_EMO: return None
        if c < CONF_MIN[e] or h < HOLD_SEC[e]: return None
        if now - s._last_nudge < COOLDOWN or e == s._last_kind: return None
        s._last_nudge, s._last_kind = now, e
        return e


def describe(st):
    """One line for the prompt. Second person for Roy, third for strangers."""
    if not st.get("seen"): return ""
    e, c = st["emotion"], st["conf"]
    if c < 0.45:   look = "expression unclear"
    elif c < 0.60: look = f"maybe {e}"
    else:          look = e
    mins = st["present"]/60
    dur = f", on camera {mins:.0f} min" if mins >= 2 else ""
    if st["person"] == "unknown":
        return f"Someone you don't recognize is on camera. They look {look} ({c:.2f}){dur}."
    return f"You can see {st['person']} — he looks {look} ({c:.2f}){dur}."