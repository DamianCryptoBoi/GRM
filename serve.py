from io import BytesIO
from fastapi import FastAPI, Form
from processor_sv3d import GRMProcessorSV3D
import uvicorn
from diffusers import AutoPipelineForText2Image
import torch
import threading
import base64
from pydantic import BaseModel
from pathlib import Path
import time
import os


class SampleInput(BaseModel):
    prompt: str


class DiffUsers:
    def __init__(self):

        print("setting up model")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        self.steps = 4
        self.guidance_scale = 0.0

        self._lock = threading.Lock()
        print("model setup done")

    def generate_image(self, prompt: str):
        generator = torch.Generator(self.device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)
        image = self.pipeline(
            prompt=prompt + ", white background",
            negative_prompt="ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
            num_inference_steps=self.steps,
            generator=generator,
            guidance_scale=self.guidance_scale,
        ).images[0]
        # Create the directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        # Define the path for the new image
        image_path = f"./images/{time.time()}.png"

        # Save the image
        image.save(image_path, format="png")

        # Return the path of the saved image
        return image_path

    def sample(self, input: SampleInput):
        try:
            with self._lock:
                return self.generate_image(input.prompt)
        except Exception as e:
            print(e)
            with self._lock:
                return self.generate_image(input.prompt)


processor = GRMProcessorSV3D()
diffusers = DiffUsers()
app = FastAPI()


@app.get("/")
async def check():
    return {"status": "ok"}


@app.post("/generate")
async def generate(prompt: str = Form()):
    image_path = diffusers.sample(SampleInput(prompt=prompt))
    return processor.process(image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
