#EcoSort

## Idea & Purpose
- People often toss trash into the most convenient bins, even when labeled for landfill, recycling, or compost.
- Labels on bins (images/text) may peel, fade, or be ignored, and people may not quickly identify which items go where.
- EcoSort bins use a camera to identify the item being thrown away and signal the correct bin.
- Aim: improve waste management and raise awareness for sustainable practices.
- Target deployment: large-scale areas like school campuses, stadiums, and public spaces.

## What It Does
- Camera identifies the item as landfill, recycle, or compost.
- Sends the data via Arduino.
- OLED display guides the user: Prompts “hold item closer”, indicates “item pending”, and displays the correct category for disposal.
- Appropriate trash bin opens automatically for the item.
- System resets for the next item.

## How We Built It
- Software: Python + PyTorch for object recognition.
- Hardware: Arduino (ESP32-CAM), servo motors, OLED display.
- Structure: Wood for the bin model.

## Challenges
- Ensuring smooth communication between software and hardware components.
- Implementing object identification from webcam to ESP32.
- Determining dataset size for accurate object detection without interference from background environments.

## Accomplishments
- Ran PyTorch for image identification, achieving ~60% accuracy.
- Managed background/environment to prevent interference with object recognition.
- Built a small-scale model simulating bin opening triggered by the identified item category.

## What We Learned
- Using Python and PyTorch for real-time object recognition.
- Connecting Python code with ESP32-CAM and sending data to Arduino for hardware control.
- Integrating software and hardware for functional prototypes.

## What's Next For EcoSort
- Train for more objects or adopt methods supporting larger datasets.
- Expand disposal categories (e.g., electronics, glass).
- Scale up to full-size, lifelike bins.
- Integrate Wi-Fi communication to remove reliance on a laptop for camera communication.
