import time
import os
import torch
from datetime import datetime
from PIL import Image
from lavis.models import load_model_and_preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

with open("./memory/description.txt", "w") as clear_file:
    clear_file.write("")

while True:
    # set up image path
    image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")
    raw_image = Image.open(image_path).convert("RGB")
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # describe current frame: overwrite description.txt
    description = model.generate({"image": image})
    f = open("./memory/description.txt", "a")
    descStr = "".join(description)
    currDateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(descStr + " " + currDateTime)
    f.write("\n")
    f.close()

    # archive description
    archive = open("./memory/archive.txt", "a")
    archive.write(descStr + " " + currDateTime)
    archive.write("\n")
    archive.close()

    # wait 8 seconds
    print("📸 Saving frame.")
    time.sleep(8)
