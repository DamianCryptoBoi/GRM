import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'

import shutil
import os.path as osp
import argparse
import torch
from functools import partial
from webui.tab_text_to_img_to_3d import create_interface_text_to_img_to_3d
from webui.tab_img_to_3d import create_interface_img_to_3d
from webui.tab_instant3d import create_interface_instant3d
from webui.runner_mod import GRMRunner
import fastapi
import time
import uvicorn
import random
from io import BytesIO
import base64
import requests
from fastapi import FastAPI, Depends, Form
import replicate
from PIL import Image

app = fastapi.FastAPI()
grm = GRMRunner(torch.device('cuda'))

def get_random_seed():
    return random.randint(0, 10000000)

def load_image_from_url(input_image: str) -> Image.Image:
    response = requests.get(input_image)
    response.raise_for_status()  # Raise an exception for HTTP errors
    image = Image.open(BytesIO(response.content))
    return image

@app.post("/generate")
async def generate(prompt: str = Form()):
    start_time = time.time()
    img_url = replicate.run("black-forest-labs/flux-schnell",
        input={
            "prompt": "3D icon of a " +prompt+", white background",
        })[0]
    print("Image URL: ", img_url)
    img = grm.run_segmentation(load_image_from_url(img_url))
    print("Segmentation completed")
    gs_path = grm.run_img_to_3d(seed=get_random_seed(),image=img)
    # gs_path = grm.run_instant3d(seed=get_random_seed(), prompt=prompt)
    print("GS Path: ", gs_path)
    # read gs model from file to buffer
    buffer = BytesIO()
    with open(gs_path, 'rb') as file:
        buffer.write(file.read())
    buffer.seek(0)
    buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    response = requests.post("http://localhost:8094/validate_ply/", json={"prompt": prompt, "data": buffer, "data_ver":1})
    score = response.json().get("score", 0)
    print(f"Prompt: {prompt.strip()}, Score: {score}")
    print(response.json())
    end_time = time.time()
    print("Time taken: ", end_time-start_time)
    return score


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8193)