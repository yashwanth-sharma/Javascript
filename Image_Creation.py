from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda")

prompt = "a futuristic city at sunset, digital art"
image = pipe(prompt).images[0]
image.save("output.png")
