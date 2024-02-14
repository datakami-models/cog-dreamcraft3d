# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from cog import BasePredictor, Input, Path, BaseModel

WEIGHTS_CACHE_DIR = "/src/models"
os.environ["HF_HOME"] = WEIGHTS_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_CACHE_DIR
os.environ["HF_HUB_CACHE"] = WEIGHTS_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "false"
os.environ["TORCH_HOME"] = WEIGHTS_CACHE_DIR
os.environ["U2NET_HOME"] = WEIGHTS_CACHE_DIR
BG_REMOVAL_MODEL_PATH = Path(WEIGHTS_CACHE_DIR) / "background_removal/u2net.onnx"


import sys
import traceback
import pprint
import random
import shutil
import warnings
from zipfile import ZipFile

import utils.image_utils as image_utils
import utils.file_utils as file_utils


print("Importing pytorch lightning")
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import trimesh

print("Importing threestudio")
from threestudio.systems.base import BaseSystem
from threestudio.utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
    ProgressCallback,
)
from threestudio.utils.config import ExperimentConfig, load_config
from threestudio.utils.misc import get_rank
from threestudio.utils.typing import Optional
from threestudio import find as threestudio_find

print("Importing dreamcraft3d")
import dreamcraft3d # we're circumventing threestudio's load_custom_modules()

N_GPUS = 1

CONFIG_PATHS = {
    'nerf': "dreamcraft3d/configs/dreamcraft3d-coarse-nerf.yaml",
    'neus': "dreamcraft3d/configs/dreamcraft3d-coarse-neus.yaml",
    'geometry': "dreamcraft3d/configs/dreamcraft3d-geometry.yaml",
    'texture': "dreamcraft3d/configs/dreamcraft3d-texture.yaml",
}

CONFIG_PATHS_FAST = {
    'nerf': "dreamcraft3d/configs/dreamcraft3d-coarse-nerf-fast.yaml",
    'neus': "dreamcraft3d/configs/dreamcraft3d-coarse-neus-fast.yaml",
    'geometry': "dreamcraft3d/configs/dreamcraft3d-geometry-fast.yaml",
    'texture': "dreamcraft3d/configs/dreamcraft3d-texture-fast.yaml",
    }

warnings.filterwarnings(module="controlnet_aux|torchvision", action="ignore")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # workaround: setting HF_HOME creates /src/models/ and model download fails
        if not Path("./models/models--DeepFloyd--IF-I-XL-v1.0").exists():
          shutil.rmtree("./models/")

        MODEL_FILES_MAP = {
            "DreamCraft3D": {
                "url": "https://weights.replicate.delivery/default/dreamcraft3d/models.tar",
                "cache_dir": "models/"
            }
        }

        # Download model weights if their cache directory doesn't exist
        for _, v in MODEL_FILES_MAP.items():
           if not os.path.exists(v["cache_dir"]):
             file_utils.download_and_extract(url=v["url"], dest=v["cache_dir"])


    # train model to generate 3D image with threestudio framework
    def train_model(
        self,
        config_path,
        image_path,
        prompt,
        guidance_scale,
        max_steps,
        n_gpus,
        weights,
        geometry_convert_from,
    ) -> str:
        extras = [
            f"data.image_path={image_path}",
            f"system.prompt_processor.prompt={prompt}",
            f"system.guidance.guidance_scale={guidance_scale}",
            f"trainer.max_steps={max_steps}",
            "system.guidance_3d.pretrained_model_name_or_path=./models/zero123/stable_zero123.ckpt",
            "system.guidance_3d.pretrained_config=./models/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        ]
        if weights:
            extras.append(f"system.weights={weights}")
        if geometry_convert_from:
            extras.append(f"system.geometry_convert_from={geometry_convert_from}")

        cfg: ExperimentConfig = load_config(config_path, cli_args=extras, n_gpus=n_gpus)
        dm = threestudio_find(cfg.data_type)(cfg.data)
        pprint.pprint(dict(cfg)) #debug
        system: BaseSystem = threestudio_find(cfg.system_type)(
            cfg.system, resumed=cfg.resume is not None
        )
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            ConfigSnapshotCallback(
                config_path,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        callbacks += [CustomProgressBar(refresh_rate=1)]

        trainer = Trainer(
            logger=False,
            callbacks=callbacks,
            inference_mode=False,
            accelerator="gpu",
            devices=-1,
            **cfg.trainer,
        )

        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        return cfg.trial_dir

    @torch.no_grad()
    def export_meshes(self, ckpt_path, parsed_config_path, n_gpus) -> None:
        # Initialize the model
        extras = [
            "system.exporter_type=mesh-exporter",
            f"resume={ckpt_path}",
            "system.exporter.context_type=cuda",
        ]
        cfg: ExperimentConfig = load_config(
            parsed_config_path, cli_args=extras, n_gpus=n_gpus
        )

        dm = threestudio_find(cfg.data_type)(cfg.data)
        pprint.pprint(cfg) #debug

        system: BaseSystem = threestudio_find(cfg.system_type)(cfg.system, resumed=True)
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
        trainer = Trainer(
            inference_mode=True,
            accelerator="gpu",
            devices=-1,
            **cfg.trainer,
        )

        # Load the model weights to gpu, otherwise the model will be loaded to cpu which may result in OOM issues
        ckpt = torch.load(ckpt_path, map_location="cuda:0")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

        # Generate the mesh
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)
        # Zip the mesh files
        no_of_iters = cfg.trainer["max_steps"]
        export_obj_path = os.path.join(
            cfg.trial_dir, "save", f"it{no_of_iters}-export", "model.obj"
        )

        return export_obj_path

    def predict(
        self,
        image: Path = Input(
            description="Image to generate a 3D object from.",
        ),
        prompt: str = Input(
            description="Prompt to generate a 3D object.",
        ),
        use_fast_configs: bool = Input(
            description="Use fast configuration files. This is less precise but much faster than the original configuration.",
            default=True,
        ),
        guidance_scale: float = Input(
            description="The scale of the guidance loss. Higher values will result in more accurate meshes but may also result in artifacts.",
            ge=1.0,
            le=50.0,
            default=5.0,
        ),
        num_steps: int = Input(
            description="Number of iterations to run the model for.",
            ge=1,
            le=10000,
            default=5000,
        ),
        seed: int = Input(
            description="The seed to use for the generation. If not specified, a random value will be used.",
            default=None,
        ),
    ) -> Path:

        # remove old output folders
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")

        # Set seed for all random number generators
        if seed is None:
            random.seed()  # Seed from current time
            seed = random.randint(0, 2**32 - 1)
            print(f"Using seed {seed}")
        
        pl.seed_everything(seed + get_rank(), workers=True)

        config_paths = CONFIG_PATHS
        if use_fast_configs:
          config_paths = CONFIG_PATHS_FAST

        # 0. Preprocess the image
        print("\n\nPreprocessing image...")
        image = image_utils.preprocess(
            image_path=str(image),
            model_path=BG_REMOVAL_MODEL_PATH,
            remove_bg=True,
            img_size=512,
            border_ratio=0.0,
            recenter=True,
        )

        print("\n\nRunning step 1: NeRF")
        trial_dir = self.train_model(
            config_path=config_paths['nerf'],
            image_path=image,
            prompt=prompt,
            max_steps=num_steps,
            guidance_scale=guidance_scale,
            n_gpus=N_GPUS,
            weights="",
            geometry_convert_from=""
        )
            
        # will be of the form: "outputs/dreamcraft3d-coarse-nerf/$prompt@LAST/ckpts/last.ckpt"
        ckpt_path = os.path.join(trial_dir, "ckpts", "last.ckpt")

        print("\n\nRunning step 2: NeuS")
        trial_dir = self.train_model(
            config_path=config_paths['neus'],
            image_path=image,
            prompt=prompt,
            max_steps=num_steps,
            guidance_scale=guidance_scale,
            n_gpus=N_GPUS,
            weights=ckpt_path,
            geometry_convert_from=""
        )

            
        # will be of the form: "outputs/dreamcraft3d-coarse-neus/$prompt@LAST/ckpts/last.ckpt"
        ckpt_path = os.path.join(trial_dir, "ckpts", "last.ckpt")

        print("\n\nRunning step 3: geometry refinement")
        trial_dir = self.train_model(
            config_path=config_paths['geometry'],
            image_path=image,
            prompt=prompt,
            max_steps=num_steps,
            guidance_scale=guidance_scale,
            n_gpus=N_GPUS,
            weights="",
            geometry_convert_from=ckpt_path
        )
        
        ckpt_path = os.path.join(trial_dir, "ckpts", "last.ckpt")

        print("\n\nRunning step 4: texture refinement")
        trial_dir = self.train_model(
            config_path=config_paths['texture'],
            image_path=image,
            prompt=prompt,
            max_steps=num_steps,
            guidance_scale=guidance_scale,
            n_gpus=N_GPUS,
            weights="",
            geometry_convert_from=ckpt_path
        )

        ckpt_path = os.path.join(trial_dir, "ckpts", "last.ckpt")
        parsed_config_path = os.path.join(trial_dir, "configs", "parsed.yaml")

        # Export the mesh
        print("\n\nRunning step 5: Exporting meshes")

        export_obj_path = self.export_meshes(
            ckpt_path=ckpt_path,
            parsed_config_path=parsed_config_path,
            n_gpus=N_GPUS,
        )

        # Prepare .glb file and return
        mesh = trimesh.load(export_obj_path, process=False)

        out_mesh_path = "outputs/mesh.glb"
        out_zip_path = "outputs/mesh.zip"
        
        mesh.export(out_mesh_path)
        with ZipFile(out_zip_path, 'w') as myzip:
            myzip.write(out_mesh_path)

        if os.path.exists("lightning_logs"):
            shutil.rmtree("lightning_logs")

        return Path(out_zip_path)
