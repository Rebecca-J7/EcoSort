## Idea & Purpose
- People often toss trash into the most convenient bins, even when labeled for landfill, recycling, or compost.
- Our idea was to create a smart, interactive waste classification system that assists users in disposing of items correctly.
- EcoSort utilizes a controlled image classification-based object identification system using region-of-interest extraction to improve waste management and raise awareness for sustainable practices.
- The goal was to combine computer vision, machine learning, and physical actuation to demonstrate a real-world AI-enabled system.
- This project serves as a prototype and proof-of-concept for automated waste sorting solutions.

## What It Does
- Uses a webcam to capture a live image of an object held by the user.
- A trained machine learning image classifier identifies whether the object belongs in: Compost, Recycling, Landfill
- The predicted class is sent as a string command from Python to an Arduino via USB serial communication.
- The Arduino interprets the command and opens the corresponding bin flap using servo motors.
- The system then resets by closing the bin, ready for the next item.
- The entire process happens in near real-time, creating a smooth interactive experience

## How We Built It
Software:
- Python: main control logic, webcam capture and image preprocessing
- TensorFlow / Keras: training and running a convolutional neural network (CNN), model saved as keras_model.h5
- OpenCV: real-time webcam feed. ROI (Region of Interest) cropping, visual feedback with a focus box
- NumPy: image array manipulation
- PySerial: communication between Python and Arduino over USB

Machine Learning:
- Image classification model trained on a custom dataset of waste images.
- Transfer learning used with a pre-trained CNN backbone for efficiency.
- ROI cropping used to reduce background noise and improve accuracy.
- Small prediction buffers used to stabilize results.

Hardware:
- Arduino (microcontroller)
- Servo motors: control physical bin flaps
- Webcam
- Mini wooden prototype: three labeled bins (Compost, Recycling, Landfill)

- Demonstrating: End-to-end ML pipeline, real-time inference, hardware integration potential, decision confidence handling

## Challenges
- Background interference: Busy environments caused incorrect classifications. Solved using a fixed ROI where users hold the object.
- Latency in Arduino response: Initial serial communication caused delayed servo movement. Fixed by optimizing serial reads and removing blocking code.
- Prediction instability: Single-frame predictions fluctuated. Addressed with short rolling buffers for majority voting.
- Hardware synchronization: Ensuring servos moved only when valid commands were received.
- Balancing speed vs accuracy: Needed fast reactions without sending incorrect commands.

## Accomplishments
- Successfully built a real-time AI-powered physical system.
- Integrated computer vision, machine learning, and hardware control.
- Achieved fast and reliable Python → Arduino communication.
- Created a working mechanical prototype that responds to ML predictions.
- Demonstrated a clear end-to-end pipeline from perception to action.
- Produced a system suitable for live demonstration and academic evaluation.

## What We Learned
- How to design and train an image classification model for real-world use.
- The importance of ROI cropping in improving computer vision accuracy.
- How serial communication latency affects physical systems.
- Best practices for non-blocking I/O in both Python and Arduino.
- The trade-offs between model confidence, speed, and user experience.
- How software decisions directly impact hardware behavior.
- How to debug and integrate multi-disciplinary systems (AI + electronics + mechanics).

## What's Next For EcoSort
- Train for more objects or adopt methods supporting larger datasets.
- Expand disposal categories (e.g., electronics, glass).
- Scale up to full-size, lifelike bins.
- Integrate Wi-Fi communication to remove reliance on a laptop for camera communication.
