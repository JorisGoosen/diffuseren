import torch
from diffusers import StableDiffusionPipeline

pipeImg = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
pipeImg = pipeImg.to("cuda")


def img(prompt, steps:int=50, num:int=2):
    for n in range(int(num/2)):
        result = pipeImg(prompt, num_inference_steps=steps, num_images_per_prompt=2) #, height=512, width=512, num_inference_steps=60)
        for i in range(len(result.images)):
            result.images[i].save(prompt + "_" + str(steps)  + "_" + str(i + n * 2) + ".png")



def imgBreed(prompt, steps:int=50, num:int=1):
    result = pipeImg(prompt, width=1024, height=256, num_inference_steps=steps, num_images_per_prompt=num) #, height=512, width=512, num_inference_steps=60)
    for i in range(len(result.images)):
        result.images[i].save(prompt + "_" + str(steps)  + "_" + str(i) + ".png")