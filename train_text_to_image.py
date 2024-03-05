#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import webdataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers import VQModel, DDIMScheduler, LDMPipeline, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import trainer_util
from trainer_util import *

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=args.sampling_steps, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--webdataset_urls",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_lion", action="store_true", help="Whether or not to use LION optimizer."
    )
    parser.add_argument(
        "--use_scram", action="store_true", help="Whether or not to use SCRAM optimizer."
    )
    parser.add_argument(
        "--use_simon", "--use_sdm", action="store_true", help="Whether or not to use SIMON optimizer."
    )
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="adam", 
        help=(
            'The optimizer type to use. Choose between ["adam", "8bit_adam", "lion", "scram", "simon", "esgd"]'
        )
    )
    parser.add_argument(
        "--rmsclip", action="store_true", help="Enable RMSclip for SIMON."
    )
    parser.add_argument(
        "--layerwise", action="store_true", help="Enable layerwise scaling for SIMON."
    )
    parser.add_argument(
        "--simon_normalize", action="store_true", help="Enable normalization for SIMON."
    )
    parser.add_argument(
        "--esgd_p", type=float, default=0.5, help="Optimizer p parameter (ESGD only)."
    )
    parser.add_argument(
        "--esgd_swap_ratio", type=float, default=1.0, help="Optimizer swap_ratio parameter (ESGD only)."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.8, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--discriminator_config", 
        type=str,
        default="discriminator_config.json", 
        help="Location of config file to use when initializing a new GAN discriminator",
    )
    parser.add_argument(
        "--gan_weight", 
        type=float, 
        default=0.2, 
        required=False, 
        help="Strength of effect GAN has on training"
    )
    parser.add_argument(
        "--stabilize_d", 
        type=float, 
        default=0.2, 
        required=False, 
        help="Loss threshold below which discriminator training will be frozen to allow the generator to catch up"
    )
    parser.add_argument(
        "--stabilize_g", 
        type=float, 
        default=0.2, 
        required=False, 
        help="Loss threshold below which generator training will be frozen to allow the discriminator to catch up"
    )
    parser.add_argument(
        "--freeze_unet", action="store_true", help="Whether to freeze the unet."
    )
    parser.add_argument(
        "--ldm", action="store_true", help="Train an unconditional latent diffusion model."
    )
    parser.add_argument(
        "--sampling_steps", 
        type=int, 
        default=5, 
        required=False, 
        help="Sampling steps during training"
    )
    parser.add_argument(
        "--log_sample_steps", 
        type=int, 
        default=100, 
        required=False, 
        help=""
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None and args.webdataset_urls is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.use_8bit_adam:
        args.optimizer = '8bit_adam'
    elif args.use_lion:
        args.optimizer = 'lion'
    elif args.use_simon:
        args.optimizer = 'simon'
    elif args.use_scram:
        args.optimizer = 'scram'

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    if not args.ldm:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )
        try:
            discriminator = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="discriminator", revision=args.non_ema_revision
            )
        except:
            discriminator = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
            )
    else:
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = None
        text_encoder = None
        vae = VQModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="vqvae", revision=args.revision)
        unet = UNet2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )
        try:
            discriminator = UNet2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="discriminator", revision=args.non_ema_revision
            )
        except:
            discriminator = UNet2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
            )
            
    in_channels = unet.config.in_channels
    vae_scaling_factor = vae.config.scaling_factor
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not args.ldm:
        text_encoder.requires_grad_(False)
    
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # TODO(rossm): Handle discriminator
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            # for i, model in enumerate(models):
                # model.save_pretrained(os.path.join(output_dir, "unet"))

                # # make sure to pop weight so that corresponding model is not saved again
                # weights.pop()

        # TODO(rossm): Handle discriminator
        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            # for i in range(len(models)):
                # # pop models so that they are not loaded again
                # model = models.pop()

                # # load diffusers style into model
                # load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                # model.register_to_config(**load_model.config)

                # model.load_state_dict(load_model.state_dict())
                # del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        discriminator.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "weight_decay": args.adam_weight_decay,
        "eps": args.adam_epsilon,
    }

    # Initialize the optimizer
    if args.optimizer == 'adam':
        optimizer_cls = torch.optim.AdamW
    elif args.optimizer == '8bit_adam':
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.optimizer == 'lion':
        from lion_pytorch import Lion

        optimizer_cls = Lion
        del optimizer_kwargs["eps"]
    elif args.optimizer == 'scram':
        from scram_pytorch import Scram

        optimizer_cls = Scram
    elif args.optimizer == 'simon':
        from scram_pytorch import Simon

        optimizer_cls = Simon
        optimizer_kwargs["rmsclip"] = args.rmsclip
        optimizer_kwargs["layerwise"] = args.layerwise
        optimizer_kwargs["normalize"] = args.simon_normalize
    elif args.optimizer == 'esgd':
        from scram_pytorch import EnsembleSGD

        optimizer_cls = EnsembleSGD
        optimizer_kwargs["p"] = args.esgd_p
        optimizer_kwargs["swap_ratio"] = args.esgd_swap_ratio
    else:
        raise ValueError(f"Unknown optimizer `{args.optimizer}`")

    optimizer = optimizer_cls(unet.parameters(), **optimizer_kwargs)
    optimizer_discriminator = optimizer_cls(discriminator.parameters(), **optimizer_kwargs)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        if not args.ldm:
            examples["input_ids"] = tokenize_captions(examples)
        return examples
        
    def preprocess_one(example):
        example["pixel_values"] = train_transforms(example[image_column].convert("RGB"))
        if not args.ldm:
            example["input_ids"] = tokenize_captions({caption_column: [example[caption_column]]})[0]
        else:
            example["input_ids"] = None
        return example

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        if not args.ldm:
            input_ids = torch.stack([example["input_ids"] for example in examples])
        else:
            input_ids = None
        return {"pixel_values": pixel_values, "input_ids": input_ids}
        
    def size_check(sample):
        image = sample[image_column]
        return image.size[0] >= args.resolution and image.size[1] >= args.resolution

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    if args.webdataset_urls is not None:
        epoch_len = 100000 // (args.train_batch_size * accelerator.num_processes)
        dataset_len = None
        fixed_epoch_len = True
        image_column = args.image_column or "image"
        caption_column = args.caption_column or "txt"
        train_dataset = (
            webdataset.WebDataset(
                args.webdataset_urls,
                repeat=True,
                shardshuffle=1000,
                nodesplitter=webdataset.split_by_node,
                handler=webdataset.warn_and_continue,
            )
            .shuffle(5000)
            .decode("pil")
            .rename(**{image_column: "png;jpg;jpeg"})
            .select(size_check)
            .map(preprocess_one)
            .with_epoch(epoch_len * args.train_batch_size * accelerator.num_processes)
            #.batched(args.train_batch_size)
        )

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    else:
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        if args.image_column is None or args.caption_column is None:
            column_names = dataset["train"].column_names
        else:
            column_names = [args.image_column, args.caption_column]

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
                
        if not args.ldm:
            if args.caption_column is None:
                caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
            else:
                caption_column = args.caption_column
                if caption_column not in column_names:
                    raise valueerror(
                        f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                    )

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        epoch_len = len(train_dataloader)
        fixed_epoch_len = False
        dataset_len = len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(epoch_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    lr_scheduler_discriminator = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_discriminator,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, discriminator, optimizer_discriminator, lr_scheduler_discriminator = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, discriminator, optimizer_discriminator, lr_scheduler_discriminator
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    if not args.ldm:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if not fixed_epoch_len:
        epoch_len = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(epoch_len / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if dataset_len != None:
        logger.info(f"  Num examples = {dataset_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps + (resume_step if args.resume_from_checkpoint else 0)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    uncond_input_ids = tokenizer(
            [""] * args.train_batch_size, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        ).input_ids.to(text_encoder.device)

    uncond_encoder_hidden_states = text_encoder(uncond_input_ids)[0]
    noise_scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(noise_scheduler.config)

    @torch.no_grad()
    def sampling_loop(sampling_steps: int, encoder_hidden_states: Tensor):
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        
        # Pick a guidance scale for each sample. We use a random guidance scale to make things more robust
        guidance_scale = torch.rand((batch_size, 1, 1, 1), device=device) * 5.0 + 5.0
        
        prompt_embeds = torch.cat([encoder_hidden_states, uncond_encoder_hidden_states])
        
        # Get timesteps for the sampling loop
        noise_scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = noise_scheduler.timesteps
        
        # Get random initial latents
        latents_size = (batch_size, in_channels, args.resolution // 8, args.resolution // 8)
        latents = torch.randn(latents_size, device=device)
        latents = latents * noise_scheduler.init_noise_sigma

        input_latents = torch.zeros((sampling_steps, *latents_size), device=device)
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            input_latents[i] = latents
            
            latent_model_input = torch.cat([latents, latents])
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = unet(latent_model_input, t, prompt_embeds).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = guidance_scale * noise_pred_text + (1 - guidance_scale) * noise_pred_uncond
            
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        return input_latents, timesteps, latents
        
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Get the text embedding for conditioning
                if not args.ldm:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                else:
                    encoder_hidden_states = torch.empty([0, 0, 0])
                    
                # Sample fake images
                input_latents, timesteps, samples = sampling_loop(args.sampling_steps, encoder_hidden_states)

                # Convert real images to latent space
                if not args.ldm:
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                else:
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latents

                bsz = latents.shape[0]
                zero_timesteps = torch.zeros((bsz,), dtype=torch.long, device=latents.device)

                # Get discriminator losses
                discriminator_output = discriminator(samples, zero_timesteps, encoder_hidden_states).sample.mean(dim=(1,2,3))
                loss_d_fake = F.binary_cross_entropy_with_logits(discriminator_output, torch.zeros_like(discriminator_output))
                loss_d_fake.backward()

                discriminator_output = discriminator(latents, zero_timesteps, encoder_hidden_states).sample.mean(dim=(1,2,3))
                loss_d_real = F.binary_cross_entropy_with_logits(discriminator_output, torch.ones_like(discriminator_output))
                loss_d_real.backward()

                # Discriminator optimization step
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                lr_scheduler_discriminator.step()

                # Get generator loss
                samples.requires_grad = True
                discriminator_output = discriminator(samples, zero_timesteps, encoder_hidden_states).sample.mean(dim=(1,2,3))
                loss_g = F.binary_cross_entropy_with_logits(discriminator_output, torch.ones_like(discriminator_output))

                # Get gradient of generator loss with respect to the sample
                loss_g.backward(inputs=(samples,))

                del discriminator_output

                # Do sample forward pass again, this time with gradient information
                sample_steps = torch.randint(0, args.sampling_steps, (bsz,), device=latents.device)
                sample_input_latents = torch.zeros_like(latents)
                for i in range(bsz):
                    sample_input_latents[i] = input_latents[sample_steps[i], i]
                generator_output = unet(sample_input_latents, timesteps[sample_steps], encoder_hidden_states).sample

                # Use the sample gradient to approximate the effect of one sampling step on the final output
                if noise_scheduler.config.prediction_type == "sample":
                    generator_output.backward(samples.grad.detach())
                else:
                    generator_output.backward(-samples.grad.detach())

                # Generator optimization step
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                del generator_output, input_latents, sample_input_latents
                
                avg_loss_d_real = accelerator.gather(loss_d_real.repeat(args.train_batch_size)).mean().detach()
                avg_loss_d_fake = accelerator.gather(loss_d_fake.repeat(args.train_batch_size)).mean().detach()
                avg_loss_g = accelerator.gather(loss_g.repeat(args.train_batch_size)).mean().detach()
                
            logs = {
                "d_loss": (avg_loss_d_real.item() + avg_loss_d_fake.item()),
                "g_loss": avg_loss_g.item(),
                "d_lr": lr_scheduler_discriminator.get_last_lr()[0],
                "g_lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                if global_step % args.log_sample_steps == 0:
                    torch.cuda.empty_cache()
                    images = []
                    with torch.no_grad():
                        for sample in samples:
                            images.append(vae.decode(sample[None,:,:,:] / vae_scaling_factor))
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "sample": [
                                        wandb.Image(image, caption=f"{i}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )
                    del images
                    torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break
            if fixed_epoch_len and step + 1 >= epoch_len:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        if not args.ldm:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        else:
            pipeline = LDMPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vqvae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)
        discriminator.save_pretrained(os.path.join(args.output_dir, "discriminator"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
