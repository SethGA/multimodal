'''capture mp4'''
import time
import os
import cv2
from PIL import Image
import numpy as np

# Folder
folder = "frames"

# Create the frames folder if it doesn't exist
frames_dir = os.path.join(os.getcwd(), folder)
os.makedirs(frames_dir, exist_ok=True)

# Initialize video
name = "beach"
cap_mp4 = cv2.VideoCapture(f'mp4/{name}.mp4')

# Check if video is opened correctly
if not cap_mp4.isOpened():
    raise IOError("Unable to load video file")

# Keep track of time
start_time = time.time()

count = 0
while True:
    ret, frame = cap_mp4.read()
    if ret and time.time() - start_time >= 2:
        # Convert the frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image
        max_size = 250
        ratio = max_size / max(pil_img.size)
        new_size = tuple([int(x*ratio) for x in pil_img.size])
        resized_img = pil_img.resize(new_size, Image.LANCZOS)

        # Convert the PIL image back to an OpenCV image
        frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)

        # Save the frame as an image file
        print("📸 Saving frame.")
        path = f"{folder}/{name}_frame_{count}.jpg"
        cv2.imwrite(path, frame)
        count += 1

        # Update start time
        start_time = time.time()

    # else:
    #     print("Failed to capture image")

    # Wait a short time
    # div video length into 5 chunks
    # hmm... dance -> 0.025
    time.sleep(.003)
    if count == 5:
        break

# Realease camera and close all windows
cap_mp4.release()
cv2.destroyAllWindows()
