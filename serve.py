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

app = fastapi.FastAPI()
grm = GRMRunner(torch.device('cuda'))


start_time = time.time()
img = grm.run_text_to_img(seed=69420,h=512, w=512, prompt='a wooden carving of a wise old turtle, best quality, sharp focus, photorealistic, extremely detailed', negative_prompt='worst quality, low quality, depth of field, blurry, out of focus, low-res, illustration, painting, drawing', steps=20, cfg_scale=7)
img = grm.run_segmentation(img)
gs = grm.run_img_to_3d(seed=69420, image=img, model="Zero123++ v1.2",cache_dir="/data")
end_time = time.time()
print("Time taken: ", end_time-start_time)
print(gs)
time.sleep(100)


