import time
import os
import cv2
import torch
from datetime import datetime
from PIL import Image


def describe_image(image_path):
    # initialize BLIP-caption
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = Image.open(image_path).convert("RGB")
    from lavis.models import load_model_and_preprocess
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    description = model.generate({"image": image})
    f = open("description.txt", "a")
    descStr = "".join(description)
    currDateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(descStr + " " + currDateTime)
    f.write("\n")
    f.close()
    return description


while True:
    # path to the current frame
    image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")
    cv2.imwrite(image_path, "frame.jpg")
    description = describe_image(image_path)
    print(description)

    # wait 8 seconds
    time.sleep(8)
