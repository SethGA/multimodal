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
cap_mp4 = cv2.VideoCapture('mp4/videoplayback.mp4')

# Check if video is opened correctly
if not cap_mp4.isOpened():
    raise IOError("Unable to load video file")

while True:
    ret, frame = cap_mp4.read()
    if ret:
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
        path = f"{folder}/frame.jpg"
        cv2.imwrite(path, frame)
    else:
        print("Failed to capture image")

    # Wait 2 seconds
    time.sleep(2)

# Realease camera and close all windows
cap_mp4.release()
cv2.destroyAllWindows()
