
from PIL import Image
from datetime import datetime
import torch;
import sympy;
from diffusers import StableDiffusionPipeline;


#device = "cuda"
device = "mps"

pipeImg = StableDiffusionPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", 
	use_auth_token=True
).to(device)


def img(prompt, neg=None, steps:int=50, num:int=1):
    bestandsnaam = prompt + "_" + (neg or "NoNegging" ) + "_"
    if len(bestandsnaam) > 220:
        bestandsnaam = bestandsnaam[0:220]
    for i in range(num):
        result = pipeImg(prompt, num_inference_steps=steps, num_images_per_prompt=1, negative_prompt=neg)
        result.images[0].save(bestandsnaam + str(steps)  + "_" + str(i) + "_" + "{:%y-%m-%d %H:%M}".format(datetime.now()) + ".png")

