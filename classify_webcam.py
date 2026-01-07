import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from queue import Queue
from threading import Thread
import serial
import serial.tools.list_ports
import time

# ---------------------------
# Arduino Setup
# ---------------------------
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return "COM9"  # fallback

arduino_port = find_arduino_port()
baud_rate = 230400 # 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=0.01)
time.sleep(2)
print(f"Connected to Arduino on {arduino_port}")

send_queue = Queue()

def arduino_worker():
    while True:
        msg = send_queue.get()
        try:
            ser.write((msg + "\n").encode())
            print(f"Sent to Arduino: {msg}")
        except Exception as e:
            print(f"Failed to send to Arduino: {e}")

Thread(target=arduino_worker, daemon=True).start()

# ---------------------------
# Load Keras Model
# ---------------------------
model = tf.keras.models.load_model("keras_model.h5")
print("Model loaded successfully")

classes = ["Compost", "Landfill", "Recycling"]

# ---------------------------
# Settings
# ---------------------------
FRAME_SIZE = 224
BOX_SIZE = 450
BUFFER_SIZE = 3        # Small buffer for quick majority voting
CONF_THRESHOLD = 0.7

# ---------------------------
# Webcam Setup
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pred_buffer = deque(maxlen=BUFFER_SIZE)
last_label_sent = "None"

print("Starting webcam classification... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # --- Single Box for User ---
    x1, y1 = cx - BOX_SIZE // 2, cy - BOX_SIZE // 2
    x2, y2 = cx + BOX_SIZE // 2, cy + BOX_SIZE // 2
    roi = frame[y1:y2, x1:x2]

    # --- Preprocess ---
    img = cv2.resize(roi, (FRAME_SIZE, FRAME_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # --- Prediction ---
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    conf = preds[idx]
    label = classes[idx] if conf >= CONF_THRESHOLD else "Uncertain"
    pred_buffer.append(label)

    # --- Immediate Majority Voting ---
    if len(pred_buffer) == BUFFER_SIZE:
        majority_label = max(set(pred_buffer), key=pred_buffer.count)

        # Send immediately if label changed and is valid
        if majority_label != "Uncertain" and majority_label != last_label_sent:
            send_queue.put(majority_label)
            last_label_sent = majority_label
            print(f"Detected: {majority_label}")

    # --- Draw Box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame,
                "Place object inside this box",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)
    cv2.putText(frame,
                last_label_sent,
                (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2)

    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
ser.close()
print("Exited cleanly")