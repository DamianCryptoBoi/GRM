from third_party.generative_models.scripts.sampling.simple_video_sample import sample as sv3d_pipe
from third_party.generative_models.scripts.sampling.simple_video_sample import build_sv3d_model
from third_party.generative_models.instant3d import build_instant3d_model, instant3d_pipe
import os
import random
import rembg
import sys
from PIL import Image
import numpy as np
import torch
import imageio
import math
import cv2
import fast_simplification
import fpsample
import open3d as o3d
from tqdm import tqdm
from utils import saveload_utils
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from webui.base_mesh import Mesh
from webui.base_mesh_renderer import MeshRenderer

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_folder}/third_party/generative_models')


# set gpu
torch.cuda.set_device(0)
device = torch.device(0)

mesh_renderer = MeshRenderer(
    near=0.01,
    far=100,
    ssaa=1,
    texture_filter='linear-mipmap-linear').to(device)


def dump_video(image_sets, path, **kwargs):
    video_out = imageio.get_writer(path, mode='I', fps=30, codec='libx264')
    for image in image_sets:
        video_out.append_data(image)
    video_out.close()


def generate_cameras(r, num_cameras=20, device='cuda:0', pitch=math.pi/8, use_fibonacci=False):
    def normalize_vecs(vectors): return vectors / \
        (torch.norm(vectors, dim=-1, keepdim=True))

    t = torch.linspace(0, 1, num_cameras).reshape(-1, 1)

    pitch = torch.zeros_like(t) + pitch

    directions = 2*math.pi
    yaw = math.pi
    yaw = directions*t + yaw

    if use_fibonacci:
        cam_pos = fibonacci_sampling_on_sphere(num_cameras)
        cam_pos = torch.from_numpy(cam_pos).float().to(device)
        cam_pos = cam_pos * r
    else:
        z = r*torch.sin(pitch)
        x = r*torch.cos(pitch)*torch.cos(yaw)
        y = r*torch.cos(pitch)*torch.sin(yaw)
        cam_pos = torch.stack(
            [x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                             device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                             dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                           dim=-1))
    rotate = torch.stack(
        (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def fibonacci_sampling_on_sphere(num_samples=1):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def generate_input_camera(r, poses, device='cuda:0', fov=50):
    def normalize_vecs(vectors): return vectors / \
        (torch.norm(vectors, dim=-1, keepdim=True))
    poses = np.deg2rad(poses)
    poses = torch.tensor(poses).float()
    pitch = poses[:, 0]
    yaw = poses[:, 1]

    z = r*torch.sin(pitch)
    x = r*torch.cos(pitch)*torch.cos(yaw)
    y = r*torch.cos(pitch)*torch.sin(yaw)
    cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                             device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                             dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                           dim=-1))
    rotate = torch.stack(
        (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix

    fx = 0.5/np.tan(np.deg2rad(fov/2))
    fxfycxcy = torch.tensor(
        [fx, fx, 0.5, 0.5], dtype=rotate.dtype, device=device)

    return cam2world, fxfycxcy


def build_grm_model(model_path):
    latest_checkpoint_file, _ = saveload_utils.load_checkpoint(
        model_path, model=None)
    model_config = latest_checkpoint_file['config'].model_config
    from model import model
    model = model.GRM(model_config).to(device).eval()
    _ = saveload_utils.load_checkpoint(latest_checkpoint_file, model=model)
    return model, model_config


def save_gaussian(latent, gs_path, model, opacity_thr=None):
    xyz = latent['xyz'][0]
    features = latent['feature'][0]
    opacity = latent['opacity'][0]
    scaling = latent['scaling'][0]
    rotation = latent['rotation'][0]

    if opacity_thr is not None:
        index = torch.nonzero(opacity.sigmoid() > opacity_thr)[:, 0]
        xyz = xyz[index]
        features = features[index]
        opacity = opacity[index]
        scaling = scaling[index]
        rotation = rotation[index]

    pc = model.gs_renderer.gaussian_model.set_data(xyz.to(torch.float32), features.to(
        torch.float32), scaling.to(torch.float32), rotation.to(torch.float32), opacity.to(torch.float32))
    pc.save_ply(gs_path)

    return gs_path


def images2gaussian(images, c2ws, fxfycxcy, model, gs_path, video_path, mesh_path=None, fuse_mesh=False, optimize_texture=False):

    if fuse_mesh:
        fib_camera_path = generate_cameras(
            r=2.9, num_cameras=200, pitch=np.deg2rad(20), use_fibonacci=True)

    camera_path = generate_cameras(
        r=2.7, num_cameras=120, pitch=np.deg2rad(20))

    with torch.no_grad():
        with torch.cuda.amp.autocast(
                enabled=True,
                dtype=torch.bfloat16
        ):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            c2ws = c2ws.to(device, dtype=torch.float32, non_blocking=True)
            fxfycxcy = fxfycxcy.to(
                device, dtype=torch.float32, non_blocking=True)

            camera_feature = torch.cat([c2ws.flatten(-2, -1), fxfycxcy], -1)
            gs, _, _ = model.forward_visual(
                images, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws)

            filter_mask = torch.nonzero((gs['xyz'].abs() < 1).sum(dim=-1) == 3)
            for key in gs:
                if key == 'depth':
                    continue
                if gs[key] is not None:
                    gs[key] = gs[key][filter_mask[:, 0],
                                      filter_mask[:, 1]].unsqueeze(0)

            return save_gaussian(gs, gs_path, model, opacity_thr=0.05)


def pad_image_to_fit_fov(image, new_fov, old_fov):
    img = Image.fromarray(image)

    scale_factor = math.tan(np.deg2rad(new_fov/2)) / \
        math.tan(np.deg2rad(old_fov/2))

    # Calculate the new size
    new_size = (int(img.size[0] * scale_factor),
                int(img.size[1] * scale_factor))

    # Calculate padding
    pad_width = (new_size[0]-img.size[0]) // 2
    pad_height = (new_size[1] - img.size[1]) // 2

    # Create padding
    padding = (pad_width, pad_height, pad_width +
               img.size[0], pad_height+img.size[1])

    # Pad the image
    img_padded = Image.new(img.mode, (new_size[0], new_size[1]), color='white')
    img_padded.paste(img, padding)
    img_padded = np.array(img_padded)
    return img_padded


def sv3d_gs(
    sv3d_model,
    grm_model,
    grm_model_cfg,
    image_path,
    num_steps=30,
    cache_dir='cache',
    fuse_mesh=False,
    optimize_texture=False,
):

    video = sv3d_pipe(model=sv3d_model,
                      input_path=image_path,
                      version='sv3d_p',
                      elevations_deg=20.0,
                      azimuths_deg=[0, 10, 30, 50, 90, 110, 130, 150, 180, 200,
                                    220, 240, 270, 280, 290, 300, 310, 320, 330, 340, 350],
                      output_folder=f'{cache_dir}/sv3d')
    torch.cuda.empty_cache()

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = video[[0, 4, 8, 12]]

    mv_images = [cv2.resize(pad_image_to_fit_fov(
        image, 50, 33.8), (input_size, input_size)) for image in mv_images]

    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images/255 - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(
        2.7, [[20, 225], [20, 225+90], [20, 225+180], [20, 225+270]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    file_path = images2gaussian(
        images, c2ws, fxfycxcy, grm_model, f'./{cache_dir}/{name}_gs.ply', f'{cache_dir}/{name}.mp4', f'{cache_dir}/{name}_mesh.ply', fuse_mesh=fuse_mesh, optimize_texture=optimize_texture)
    torch.cuda.empty_cache()
    return file_path


class GRMProcessorSV3D:
    def __init__(self) -> None:
        print("Init models")
        self.grm_uniform_path = 'checkpoints/grm_u.pth'
        self.grm_uniform_model, self.grm_uniform_config = build_grm_model(
            self.grm_uniform_path)

        self.sv3d_model = build_sv3d_model(
            num_steps=30,
            device=device,
        )
        print("Done")

    def process(self, img_path: str, steps: int = 30):
        return sv3d_gs(
            sv3d_model=self.sv3d_model,
            grm_model=self.grm_uniform_model,
            grm_model_cfg=self.grm_uniform_config,
            image_path=img_path,
            num_steps=30,
        )
