## Idea & Purpose
- People often toss trash into the most convenient bins, even when labeled for landfill, recycling, or compost.
- Labels on bins (images/text) may peel, fade, or be ignored, and people may not quickly identify which items go where.
- EcoSort bins use a camera to identify the item being thrown away and signal the correct bin.
- Aim: improve waste management and raise awareness for sustainable practices.
- Target deployment: large-scale areas like school campuses, stadiums, and public spaces.

## What It Does
- The web camera identifies the item as landfill, recycling, or compost.
- Data collected by the camera is processed through a calculated average and then sent via Arduino.
- OLED display guides the user: Prompts “hold item closer”, indicates “hold item still”, and displays the correct category for disposal.
- After the data is processed, the appropriate trash bin opens automatically for the item, and the system resets for the next item.

## How We Built It
- Software: Python + PyTorch for object recognition.
- Hardware: Arduino, servo motors, OLED display.
- Structure: Wood for the bin model.

## Challenges
- Ensuring smooth communication between software and hardware components.
- Implementing object identification from a laptop webcam.
- Determining the dataset size for accurate object detection without interference from background environments.
- Determining whether to accommodate identification for multiple items, though we decided to shift focus to identifying one item at a time.

## Accomplishments
Ran PyTorch for image identification.
Managed background/environment to prevent some interference with object recognition.
Built a small-scale model simulating bin opening triggered by the identified item category.

## What We Learned
- Using Python and PyTorch for real-time object recognition through image transformation.
- Connecting Python code with ESP32-CAM and sending data to Arduino for hardware control.
- Integrating software and hardware for functional prototypes. (laptop <-> webcamera, webcamera + code<->arduino, arduino<->bins)
- Creating our model with digital servos has higher torque for bin flap movement, and refining our design to be simple and take up less space & resources.

## What's Next For EcoSort
- Train for more objects or adopt methods supporting larger datasets.
- Expand disposal categories (e.g., electronics, glass).
- Scale up to full-size, lifelike bins.
- Integrate Wi-Fi communication to remove reliance on a laptop for camera communication.
