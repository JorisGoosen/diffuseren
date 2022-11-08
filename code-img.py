
from PIL import Image

from diffusers import StableDiffusionPipeline;


#device = "cuda"
device = "mps"

pipeImg = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to(device)


def img(prompt, steps:int=50, num:int=1):
    result = pipeImg(prompt, num_inference_steps=steps, num_images_per_prompt=num) #, height=512, width=512, num_inference_steps=60)
    for i in range(len(result.images)):
        result.images[i].save(prompt + "_" + str(steps)  + "_" + str(i) + ".png")

