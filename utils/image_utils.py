import os
from pathlib import Path
from rembg import remove

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from dreamcraft3d.dpt import DPTDepthModel


class DPT:
    def __init__(self, task="depth", device="cuda"):
        self.task = task
        self.device = device

        if task == "depth":
            path = hf_hub_download(
                repo_id="clay3d/omnidata", filename="omnidata_dpt_depth_v2.ckpt"
            )
            self.model = DPTDepthModel(backbone="vitb_rn50_384")
            self.aug = transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )

        else:  # normal
            path = hf_hub_download(
                repo_id="clay3d/omnidata", filename="omnidata_dpt_normal_v2.ckpt"
            )
            self.model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
            self.aug = transforms.Compose(
                [transforms.Resize((384, 384)), transforms.ToTensor()]
            )

        # load model
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == "depth":
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
            )
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(
                normal, size=(H, W), mode="bicubic", align_corners=False
            )
            normal = normal.cpu().numpy()
            return normal


def preprocess(image_path,
               model_path,
               remove_bg=True, 
               img_size=512, 
               border_ratio=0.2, 
               recenter=True
):

    if not Path("/src/outputs").exists():
        os.mkdir("/src/outputs")
    out_rgba = os.path.join('/src/outputs/image_rgba.png')
    out_normal = os.path.join('/src/outputs/image_normal.png')
    out_depth = os.path.join('/src/outputs/image_depth.png')

    # load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] != 4 and remove_bg == False:
        raise ValueError("Please provide an RGBA image with background removed or set remove_bg=True.")
    
    # RGBA image of shape [height, width, 4]
    if remove_bg:
        print(f'[INFO] background removal...')
        carved_image = remove(image, model="u2net_custom", model_path=model_path)
    else:
        carved_image = image

    mask = carved_image[..., -1] > 0

    # DPT expects an image without alpha channel, so image.shape == 3
    # if the loaded image already has an alpha channel, we throw that info away
    if image.shape[2] == 4:
      image = image[:,:,:3]
    
    # predict depth
    print(f"[INFO] depth estimation...")
    dpt_depth_model = DPT(task="depth")
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (
        2 * depth[mask].mean() - depth[mask].min() + 1e-9
    )
    depth[~mask] = 0
    depth[depth > 1.0] = 1.0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model

    # predict normal
    print(f"[INFO] normal estimation...")
    dpt_normal_model = DPT(task="normal")
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model

    # Recenter image
    if recenter:
        final_rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)
        final_depth = np.zeros((img_size, img_size), dtype=np.uint8)
        final_normal = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        height = x_max - x_min
        width = y_max - y_min
        desired_size = int(img_size * (1 - border_ratio))
        scale = desired_size / max(height, width)

        height_new = int(height * scale)
        width_new = int(width * scale)
        x2_min = (img_size - height_new) // 2
        x2_max = x2_min + height_new
        y2_min = (img_size - width_new) // 2
        y2_max = y2_min + width_new
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            carved_image[x_min:x_max, y_min:y_max], 
            (width_new, height_new), 
            interpolation=cv2.INTER_AREA
        )
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            depth[x_min:x_max, y_min:y_max], (width_new, height_new), interpolation=cv2.INTER_AREA
        )
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            normal[x_min:x_max, y_min:y_max], (width_new, height_new), interpolation=cv2.INTER_AREA
        )
    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal
        
    # write image
    cv2.imwrite(out_rgba, final_rgba)
    cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)
    return out_rgba
