import math
import torch
import torch.nn.functional as F
import einops
from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch import Tensor
from torch.nn import Module

def unsqueeze_like(x: Tensor, target: Tensor):
    x = x.flatten()
    while len(x.shape) < len(target.shape):
        x = x.unsqueeze(-1)
    return x

def get_predicted_latents(
        noisy_latents: Tensor, 
        model_pred: Tensor, 
        timesteps: Tensor, 
        noise_scheduler: DDPMScheduler,
    ):
    """
    Computes the target latents for another timestep based on the the generator's input and output.
    """
    original_dtype = noisy_latents.dtype
    noisy_latents = noisy_latents.float()
    model_pred = model_pred.float()
    
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = unsqueeze_like(sqrt_alpha_prod, noisy_latents)
        
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = unsqueeze_like(sqrt_one_minus_alpha_prod, noisy_latents)
    
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_latents = (noisy_latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod
        #predicted_noise = model_pred
    elif noise_scheduler.config.prediction_type == "v-prediction":
        predicted_latents = sqrt_alpha_prod * noisy_latents - sqrt_one_minus_alpha_prod * model_pred
        #predicted_noise = sqrt_one_minus_alpha_prod * noisy_latents + sqrt_alpha_prod * model_pred
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return predicted_latents.to(dtype=original_dtype)

def batch_repeat(x: Tensor, count: int):
    if count == 1:
        return x
    return einops.repeat(x, 'b ... -> (c b) ...', c=count)

def log_grad_norm(model_name: str, model: Module, accelerator: Accelerator, global_step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / math.sqrt(grads.numel())).item()
            accelerator.log({f"grad_norm/{model_name}/{name}": grad_norm}, step=global_step)
            values = param.detach().data
            value_norm = (values.norm(p=2) / math.sqrt(values.numel())).item()
            accelerator.log({f"value_norm/{model_name}/{name}": value_norm}, step=global_step)
