
from PIL import Image

from diffusers import StableDiffusionPipeline;


device = "mps"
model_path = "CompVis/stable-diffusion-v1-4"

pipeImg = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to(device)


def img(prompt, steps=50):
    result = pipeImg(prompt, num_inference_steps=steps) #, height=512, width=512, num_inference_steps=60)
    result.images[0].save(prompt + "_" + str(steps) + ".png")


