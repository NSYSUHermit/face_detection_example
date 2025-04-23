from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
import threading
import queue
import uuid
from deepface import DeepFace

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

DETECT_INTERVAL = 15
CONF_THRESHOLD = 0.4
SMOOTH_ALPHA = 0.5
KNOWN_FACES_DIR = "known_faces"
FRAME_QUEUE = queue.Queue()
RESULT_LOCK = threading.Lock()
COLORS = {}
FONT = cv2.FONT_HERSHEY_SIMPLEX

cached_boxes = []
cached_labels = []
prev_boxes = []
frame_count = 0

EMBEDDING_CACHE = {}
FACENET_MODEL = DeepFace.build_model("Facenet")

# 🛠 修正 DeepFace.represent 調用，移除不支援的 'model' 參數
def preload_embeddings():
    print("🔍 載入 known_faces 特徵向量...")
    for file in os.listdir(KNOWN_FACES_DIR):
        ref_path = os.path.join(KNOWN_FACES_DIR, file)
        if not os.path.isfile(ref_path):
            continue
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"⚠️ 跳過非圖像檔案: {file}")
            continue
        try:
            embedding = DeepFace.represent(
                img_path=ref_path,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip"
            )[0]["embedding"]
            EMBEDDING_CACHE[ref_path] = (os.path.splitext(file)[0], embedding)
            print(f"✅ {file} 載入成功")
        except Exception as e:
            print(f"❌ {file} 載入失敗: {e}")

preload_embeddings()

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_by_embedding(face_img):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, face_img)

    try:
        target_embedding = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="skip"
        )[0]["embedding"]
    except Exception as e:
        os.remove(temp_path)
        return "Unknown", 0.0

    os.remove(temp_path)
    best_match = "Unknown"
    best_score = -1
    for name, ref_embedding in EMBEDDING_CACHE.values():
        sim = cosine_similarity(target_embedding, ref_embedding)
        if sim > best_score:
            best_score = sim
            best_match = name

    if best_score < 0.4:
        return "Unknown", best_score
    return best_match, best_score

def smooth_boxes(new_boxes, prev_boxes, alpha=0.5):
    if not prev_boxes or len(prev_boxes) != len(new_boxes):
        return new_boxes
    return [(int(alpha * x1 + (1-alpha) * px1),
             int(alpha * y1 + (1-alpha) * py1),
             int(alpha * x2 + (1-alpha) * px2),
             int(alpha * y2 + (1-alpha) * py2))
            for (x1, y1, x2, y2), (px1, py1, px2, py2) in zip(new_boxes, prev_boxes)]

def inference_thread():
    global cached_boxes, cached_labels, prev_boxes, frame_count
    while True:
        if FRAME_QUEUE.empty():
            time.sleep(0.01)
            continue
        frame = FRAME_QUEUE.get()
        frame_count += 1
        if frame_count % DETECT_INTERVAL != 0:
            continue
        boxes, labels = [], []
        try:
            results = model.predict(source=frame, imgsz=128, classes=0, conf=CONF_THRESHOLD, stream=False)
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None or r.boxes.xyxy is None:
                    continue
                for box in r.boxes.xyxy:
                    if len(box) != 4:
                        continㄠue
                    x1, y1, x2, y2 = map(int, box)
                    face_img = frame[y1+5:y2-5, x1+5:x2-5]
                    name, confidence = recognize_by_embedding(face_img)
                    boxes.append((x1, y1, x2, y2))
                    labels.append((name, confidence))
        except Exception as e:
            print(f"推論錯誤: {e}")
            continue
        with RESULT_LOCK:
            boxes = smooth_boxes(boxes, prev_boxes, alpha=SMOOTH_ALPHA)
            cached_boxes = boxes
            cached_labels = labels
            prev_boxes = boxes

# 啟動推論背景執行緒
t = threading.Thread(target=inference_thread, daemon=True)
t.start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    FRAME_QUEUE.put(frame.copy())

    with RESULT_LOCK:
        for ((x1, y1, x2, y2), (label, conf)) in zip(cached_boxes, cached_labels):
            if label not in COLORS:
                COLORS[label] = tuple(np.random.randint(0, 255, size=3).tolist())
            color = COLORS[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf*100:.1f}%)"
            text_size, _ = cv2.getTextSize(text, FONT, 0.7, 2)
            text_w, text_h = text_size
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), FONT, 0.7, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 + DeepFace 即時辨識", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
