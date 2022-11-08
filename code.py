
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
import os

import shutil
from datetime import datetime

device = "mps"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
)
pipe = pipe.to(device)

from PIL import Image

import colorsys
import re
import torchvision

def img2img(file, prompt, strength=0.5, guidance_scale=7.5, alternative="", targetFile="", steps=40, dimming=1.0, hueing=0.0):
    init_img = Image.open(file).convert("RGB")
    init_img = init_img.resize((512, 512))
    init_img = torchvision.transforms.functional.adjust_hue(init_img, hueing)
    init_img = torchvision.transforms.functional.adjust_saturation(init_img, dimming)
    #init_img = torchvision.transforms.functional.(init_img, 0.1)
    result = pipe(prompt=prompt, init_image=init_img, strength=strength, guidance_scale=guidance_scale, num_inference_steps=steps)
    image = result.images[0]
    nsfw = result.nsfw_content_detected[0]
    if not nsfw:
        image.save(targetFile if targetFile != "" else (alternative if alternative != "" else file) + "_" + prompt + "_"+str(strength)+".png")
    else:
        print("NSFW")
    return not nsfw



def dream(file, prompt, strength=0.75, steps=40, dimming=1.0, dirname="", hueing=0.0):
    if dirname == "":
        now = str(datetime.now())
        dirname = prompt + now.replace("/", "-")
    counter = 0
    wasntDirThere = not os.path.exists(dirname)
    if wasntDirThere:
        print("dir doesnt exist")
        os.mkdir(dirname)
    else:
        print("dir exists")
        previous = os.listdir(dirname)
        previous.sort(reverse=True)
        m = re.search("(\d+)\.png", previous[0])
        counter = int(m.group(1))
        print("Found counter " + str(counter))
    def counterFile(next=False):
        return  dirname + "/%06d.png" % (counter + 1 if next else counter)
    if wasntDirThere:
        shutil.copyfile(file, counterFile())
    while(True):
        if img2img(file=counterFile(), prompt=prompt, strength=strength, targetFile=counterFile(True), steps=steps, dimming=dimming, hueing=hueing):
            counter = counter + 1


#dream("purple dog with wings, a drawing.png", "drawing of dog", 0.5)



init_img = Image.open("basis.png").convert("RGB")
result = pipe(prompt="whatever", init_image=init_img,num_inference_steps=1)