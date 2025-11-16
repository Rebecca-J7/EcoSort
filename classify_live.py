import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import deque
import serial
import serial.tools.list_ports
import time
from queue import Queue
from threading import Thread, Lock

# ---------------------------
# Arduino Serial Setup
# ---------------------------
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return "COM9"

arduino_port = find_arduino_port()
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=0.01)
time.sleep(2)
print(f"Connected to Arduino on {arduino_port}")

# ---------------------------
# Classes and Colors
# ---------------------------
classes = ["Compost", "Landfill", "Recycling"]
colors = {"Compost": (0,255,0), "Landfill": (0,0,255), "Recycling": (255,0,0)}

# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("waste_model.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------------
# Webcam Setup
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

roi_size = 450
inner_crop_scale = 0.55
movement_buffer = deque(maxlen=2)

# ---------------------------
# Shared Variables
# ---------------------------
latest_frame = None
frame_lock = Lock()

roi_queue = Queue(maxsize=5)
pred_buffer = deque(maxlen=5)
latest_label = "Detecting..."
confidence_threshold = 0.7

send_queue = Queue()
last_label_sent = ""

# ---------------------------
# Camera Thread
# ---------------------------
def camera_worker():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()

cam_thread = Thread(target=camera_worker, daemon=True)
cam_thread.start()

# ---------------------------
# Inference Thread
# ---------------------------
def inference_worker():
    global latest_label
    while True:
        if not roi_queue.empty():
            roi_focus = roi_queue.get()
            roi_small = cv2.resize(roi_focus, (224,224))
            img_pil = Image.fromarray(cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB))
            x_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(x_tensor)
                probs = torch.nn.functional.softmax(preds, dim=1)
                conf, idx = probs.max(dim=1)
                conf = conf.item()
                label = classes[idx.item()]
            
            if conf < confidence_threshold:
                label = "hold closer"
            
            pred_buffer.append(label)
            if len(pred_buffer) == pred_buffer.maxlen:
                latest_label = max(set(pred_buffer), key=pred_buffer.count)
            else:
                latest_label = "Detecting..."

inf_thread = Thread(target=inference_worker, daemon=True)
inf_thread.start()

# ---------------------------
# Arduino Thread
# ---------------------------
def arduino_worker():
    global last_label_sent
    while True:
        msg = send_queue.get()
        try:
            ser.write((msg+"\n").encode())
            last_label_sent = msg
            print(f"Sent to Arduino: {msg}")
        except Exception as e:
            print(f"Serial write failed: {e}")

arduino_thread = Thread(target=arduino_worker, daemon=True)
arduino_thread.start()

# ---------------------------
# Main Loop
# ---------------------------
frame_count = 0

while True:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()
    
    h, w, _ = frame.shape
    cx, cy = w//2, h//2

    # ROI and inner focus
    x1, y1 = cx - roi_size//2, cy - roi_size//2
    x2, y2 = cx + roi_size//2, cy + roi_size//2
    inner_size = int(roi_size*inner_crop_scale)
    ix1 = roi_size//2 - inner_size//2
    iy1 = roi_size//2 - inner_size//2
    ix2, iy2 = ix1 + inner_size, iy1 + inner_size

    roi_big = frame[y1:y2, x1:x2]
    roi_focus = roi_big[iy1:iy2, ix1:ix2]

    # Movement detection
    roi_gray = cv2.cvtColor(roi_focus, cv2.COLOR_BGR2GRAY)
    roi_small = cv2.resize(roi_gray, (16,16))
    movement_buffer.append(roi_small)
    still = np.mean(cv2.absdiff(movement_buffer[-1], movement_buffer[-2])) < 2.5 if len(movement_buffer)==2 else True

    # Send ROI to inference
    if frame_count % 2 == 0 and still:
        if not roi_queue.full():
            roi_queue.put(roi_focus)

    # Queue message to Arduino
    if latest_label in classes and latest_label != last_label_sent:
        send_queue.put(latest_label)

    # Display overlay
    color = colors.get(latest_label, (0,255,255))
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.rectangle(frame,(x1+ix1,y1+iy1),(x1+ix2,y1+iy2),(0,255,255),1)
    cv2.putText(frame, latest_label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Waste Classifier", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()