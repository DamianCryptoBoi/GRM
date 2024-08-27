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

app = fastapi.FastAPI()
grm = GRMRunner(torch.device('cuda'))



img = grm.run_text_to_img(seed=69420,h=512, w=512, prompt='a wooden carving of a wise old turtle, best quality, sharp focus, photorealistic, extremely detailed', negative_prompt='worst quality, low quality, depth of field, blurry, out of focus, low-res, illustration, painting, drawing', steps=20, cfg_scale=7)

print(img)

