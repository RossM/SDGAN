import torch
import torch.nn.functional as F
import einops
from diffusers import DDPMScheduler
from torch import Tensor
from discriminator import Discriminator2D

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
    
    alphas_cumprod = noise_scheduler.alphas_cumprod
    
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_latents = (noisy_latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod
        #predicted_noise = model_pred
    elif noise_scheduler.config.prediction_type == "v-prediction":
        predicted_latents = sqrt_alpha_prod * noisy_latents - sqrt_one_minus_alpha_prod * model_pred
        #predicted_noise = sqrt_one_minus_alpha_prod * noisy_latents + sqrt_alpha_prod * model_pred
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return predicted_latents.to(dtype=original_dtype)

def get_discriminator_input(
        discriminator: Discriminator2D,
        noise_scheduler: DDPMScheduler,
        noisy_latents: Tensor,
        model_pred: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ):
    # Discriminator training combined positive and negative model_pred into a single batch, so repeat
    # the other arguments until they're the right batch size
    noisy_latents = batch_repeat(noisy_latents, model_pred.shape[0] // noisy_latents.shape[0])
    timesteps = batch_repeat(timesteps, model_pred.shape[0] // timesteps.shape[0])
    noise = batch_repeat(noise, model_pred.shape[0] // noise.shape[0])
    
    if discriminator.config.prediction_type == "target":
        # In target mode, the discriminator predicts directly from the unet output
        discriminator_input = model_pred
    elif discriminator.config.prediction_type == "step":
        # In step mode, the discriminator gets the simulated result of stepping the denoising process several steps.
        next_timesteps = torch.clamp(timesteps + discriminator.config.step_offset, min=0, max=noise_scheduler.config.num_train_timesteps-1)
        predicted_latents = get_predicted_latents(noisy_latents, model_pred, timesteps, noise_scheduler)
        discriminator_input = noise_scheduler.add_noise(predicted_latents, noise, next_timesteps)
    elif discriminator.config.prediction_type == "noisy_step":
        # In noisy step mode, we take the predicted denoised latents and add new noise
        # This helps regularize the discriminator. See https://arxiv.org/abs/2206.02262
        next_timesteps = torch.clamp(timesteps + discriminator.config.step_offset, min=0, max=noise_scheduler.config.num_train_timesteps-1)
        predicted_latents = get_predicted_latents(noisy_latents, model_pred, timesteps, noise_scheduler)
        discriminator_input = noise_scheduler.add_noise(predicted_latents, torch.randn_like(predicted_latents), next_timesteps)
    else:
        raise ValueError(f"Unknown prediction type {discriminator.config.prediction_type}")
        
    if discriminator.config.in_channels > discriminator_input.shape[1]:
        # Most discriminator modes get both the unet input and output
        discriminator_input = torch.cat((noisy_latents, discriminator_input), 1)

    return discriminator_input
    
def batch_repeat(x: Tensor, count: int):
    if count == 1:
        return x
    return einops.repeat(x, 'b ... -> (c b) ...', c=count)
