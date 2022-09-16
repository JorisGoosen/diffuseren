
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline


device = "mps"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
)
pipe = pipe.to(device)

from PIL import Image


def img2img(file, prompt, strength=0.9, guidance_scale=7.5, alternative=""):
    init_img = Image.open(file).convert("RGB")
    init_img = init_img.resize((512, 512))
    image = pipe(prompt=prompt, init_image=init_img, strength=strength, guidance_scale=guidance_scale).images[0]
    image.save((alternative if alternative != "" else file) + "_" + prompt + ".png")

