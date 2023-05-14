import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops, einops.layers.torch
import diffusers
from diffusers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import Tensor
from typing import Tuple, Optional
from trainer_util import *

def Downsample(dim, dim_out):
    return nn.Conv2d(dim, dim_out, 4, 2, 1)

class Conv2dLayer(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size=3, groups=32, bias=True):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, dim, dim_out, *, groups=32, bias=True, norm=True):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim) if norm else None
        self.activation = nn.SiLU()
        self.conv = nn.Linear(dim, dim_out, bias=bias)
    
    def forward(self, x):
        if self.norm != None:
            x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, *, kernel_size=3, groups=32, time_embedding_dim=128):
        super().__init__()
        self.conv_in = Conv2dLayer(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv_out = Conv2dLayer(dim, dim, kernel_size=kernel_size, groups=groups, bias=False)
        
        nn.init.zeros_(self.conv_out.conv.weight)
        #nn.init.zeros_(self.conv_out.conv.bias)
        
        if time_embedding_dim > 0:
            self.embed_in = nn.Linear(time_embedding_dim, dim, bias=False)
            nn.init.zeros_(self.embed_in.weight)
        else:
            self.embed_in = None
    
    def forward(self, input, time_embed):
        x = self.conv_in(input)
        if self.embed_in:
            x = x + einops.rearrange(self.embed_in(time_embed), 'b c -> b c 1 1')
        x = self.conv_out(x)
        return x + input
        
class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        
        init_scale = (in_features // groups) ** -0.5
        
        self.weight = nn.Parameter(torch.empty(groups, in_features // groups, out_features // groups, device=device, dtype=dtype))
        nn.init.uniform_(self.weight, -init_scale, init_scale)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            nn.init.uniform_(self.bias, -init_scale, init_scale)
        else:
            self.bias = None

    def forward(self, x):
        x = einops.rearrange(x, '... (g i) -> ... g i', g=self.groups)
        x = torch.einsum('... g i, g i o -> ... g o', x, self.weight)
        x = einops.rearrange(x, '... g o -> ... (g o)')
        if self.bias != None:
            bias = self.bias
            while len(bias.shape) < len(x.shape):
                bias = bias.unsqueeze(0)
            x = x + bias
        return x
        
    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, {self.groups}"

class CombinedAttention(nn.Module):
    def __init__(self, dim, out_dim, *, heads=8, key_dim=32, value_dim=32, bias=True, cond_embedding_dim=None, grouped=False):
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.heads = heads
        self.key_dim = key_dim

        self.to_k = nn.Linear(dim, key_dim)
        self.to_v = nn.Linear(dim, value_dim)
        if grouped:
            self.to_q = GroupedLinear(dim, key_dim * heads, heads)
        else:
            self.to_q = nn.Linear(dim, key_dim * heads)
        self.to_out = nn.Linear(value_dim * heads, out_dim, bias=bias)
        
        if cond_embedding_dim:
            self.embed_to_k = nn.Linear(cond_embedding_dim, key_dim)
            self.embed_to_v = nn.Linear(cond_embedding_dim, value_dim)
        else:
            self.embed_to_k = self.embed_to_v = None
        
        nn.init.zeros_(self.to_out.weight)
        if bias:
            nn.init.zeros_(self.to_out.bias)
        
        self.use_efficient_attention = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, cond_embed = None):
        shape = x.shape
        x = einops.rearrange(x, 'b c ... -> b (...) c')

        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_q(x)
        
        if self.embed_to_k != None:
            k = torch.cat([k, self.embed_to_k(cond_embed)], dim=1)
            v = torch.cat([v, self.embed_to_v(cond_embed)], dim=1)
        
        q = einops.rearrange(q, 'b n (h c) -> b (n h) c', h=self.heads)
        if self.use_efficient_attention:
            result = F.scaled_dot_product_attention(q, k, v)
        else:
            attention_scores = torch.bmm(q, k.transpose(-2, -1))
            attention_probs = torch.softmax(attention_scores.float() / math.sqrt(self.key_dim), dim=-1).type(attention_scores.dtype)
            result = torch.bmm(attention_probs, v)
        result = einops.rearrange(result, 'b (n h) c -> b n (h c)', h=self.heads)
        out = self.to_out(result)

        out = einops.rearrange(out, 'b n c -> b c n')
        out = torch.reshape(out, (shape[0], self.out_dim, *shape[2:]))
        return out

class CombinedAttentionBlock(nn.Module):
    def __init__(self, dim, attention_dim, *, heads=8, groups=32, cond_embedding_dim=None, v_mult=1, qk_mult=1, grouped=False):
        super().__init__()
        
        if not attention_dim:
            attention_dim = dim // heads

        self.norm = nn.GroupNorm(groups, dim)
        self.attention = CombinedAttention(
            dim, 
            dim, 
            heads=heads, 
            key_dim=attention_dim * qk_mult, 
            value_dim=attention_dim * v_mult, 
            bias=False, 
            cond_embedding_dim=cond_embedding_dim,
            grouped=grouped,
        )

    def forward(self, input, cond_embed):
        x = self.norm(input)
        x = self.attention(x, cond_embed)
        return x + input

class SequentialWithEmbeddings(nn.Sequential):
    def forward(self, x, time_embed, cond_embed):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, time_embed)
            elif isinstance(module, CombinedAttentionBlock):
                x = module(x, cond_embed)
            else:
                x = module(x)
        return x

class AdaptiveReduce(nn.Module):
    """
    Reduces over all spatial dimensions, with a learnable parameter for each
    channel that blends between max-like, mean, and min-like reduction.
    """

    def __init__(self, dim, init_scale=0):
        super().__init__()
        self.dim = dim
        
        self.a = nn.Parameter(init_scale * torch.randn(1, dim, 1))
    
    def forward(self, x):
        x = einops.rearrange(x, 'b c ... -> b c (...)')
        weight = F.softmax(self.a * x, dim=-1)
        return torch.sum(x * weight, dim=-1)
        
    def extra_repr(self):
        return f"{self.dim}"
        
def MeanReduce():
    return einops.layers.torch.Reduce('b c ... -> b c', 'mean')
        
class Discriminator2D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        in_channels: int = 8,
        out_channels: int = 1,
        block_out_channels: Tuple[int] = (128, 256, 512, 1024, 1024, 1024),
        block_repeats: Tuple[int] = (2, 2, 2, 2, 2),
        downsample_blocks: Tuple[int] = (0, 1, 2),
        mlp_hidden_channels: Tuple[int] = (2048, 2048, 2048),
        mlp_uses_norm: bool = True,
        attention_dim: Optional[int] = None,
        attention_heads: Tuple[int] = (0, 0, 8, 8),
        groups: int = 32,
        embedding_dim: int = 768,
        time_embedding_dim: int = 128,
        reduction_type: str = "MeanReduce",
        prediction_type: str = "target",
        step_offset: int = 0,
        step_type: str = "relative",
        combined_attention: bool = False,
        grouped_attention: bool = False,
        v_mult: int = 1,
        qk_mult: int = 1,
    ):
        super().__init__()
        
        if prediction_type != "target" and prediction_type != "step" and prediction_type != "noisy_step":
            raise ValueError(f"Unknown prediction type {prediction_type}")
        if step_type != "relative" and step_type != "absolute":
            raise ValueError(f"Unknown step type {step_type}")
        
        self.blocks = nn.ModuleList([])
        self.block_means = nn.ModuleList([])
        
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 7, padding=3)
        
        for i in range(0, len(block_out_channels) - 1):
            block_in = block_out_channels[i]
            block_out = block_out_channels[i + 1]
            block = SequentialWithEmbeddings()
            for j in range(0, block_repeats[i]):
                if attention_heads[i] > 0:
                    block.append(CombinedAttentionBlock(
                        block_in, 
                        attention_dim, 
                        heads=attention_heads[i], 
                        groups=groups, 
                        cond_embedding_dim=embedding_dim if combined_attention else 0,
                        v_mult=v_mult,
                        qk_mult=qk_mult,
                        grouped=grouped_attention,
                    ))
                block.append(ResnetBlock(block_in, groups=groups, time_embedding_dim=time_embedding_dim))
            if i in downsample_blocks:
                block.append(Downsample(block_in, block_out))
            elif block_in != block_out:
                block.append(nn.Conv2d(block_in, block_out, 1))
            self.blocks.append(block)
            if reduction_type == "AdaptiveReduce":
                self.block_means.append(AdaptiveReduce(block_out))
            elif reduction_type == "AdaptiveReduceN":
                self.block_means.append(AdaptiveReduce(block_out, init_scale=2/3))
            elif reduction_type == "MeanReduce":
                self.block_means.append(MeanReduce())
            else:
                raise ValueError(f"Unknown reduction type {reduction_type}")

        # A simple MLP to make the final decision based on statistics from
        # the output of every block
        self.to_out = nn.Sequential()
        d_channels = sum(block_out_channels[1:]) + embedding_dim
        self.to_out.append(nn.Linear(d_channels, mlp_hidden_channels[0], bias=not mlp_uses_norm))
        for i in range(0, len(mlp_hidden_channels) - 1):
            mlp_in = mlp_hidden_channels[i]
            mlp_out = mlp_hidden_channels[i + 1]
            self.to_out.append(MLPLayer(mlp_in, mlp_out, bias=not mlp_uses_norm, norm=mlp_uses_norm))
        final_layer = MLPLayer(mlp_hidden_channels[-1], out_channels, bias=True, norm=mlp_uses_norm)
        nn.init.zeros_(final_layer.conv.bias)
        self.to_out.append(final_layer)
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        
    def forward(self, x, timesteps, encoder_hidden_states):
        x = self.conv_in(x)
        time_embed = get_timestep_embedding(timesteps,  self.config.time_embedding_dim)
        if self.config.embedding_dim != 0:
            d = einops.reduce(encoder_hidden_states, 'b n c -> b c', 'mean')
        else:
            d = torch.zeros([x.shape[0], 0], device=x.device, dtype=x.dtype)
        for (block, block_mean) in zip(self.blocks, self.block_means):
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, time_embed, encoder_hidden_states)
            else:
                x = block(x, time_embed, encoder_hidden_states)
            x_mean = block_mean(x)
            d = torch.cat([d, x_mean], dim=-1)
        return self.to_out(d) + 0.5

    def get_input(
        self,
        noise_scheduler: DDPMScheduler,
        noisy_latents: Tensor,
        model_pred: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ):
        # Discriminator training combines positive and negative model_pred into a single batch, so repeat
        # the other arguments until they're the right batch size
        noisy_latents = batch_repeat(noisy_latents, model_pred.shape[0] // noisy_latents.shape[0])
        timesteps = batch_repeat(timesteps, model_pred.shape[0] // timesteps.shape[0])
        noise = batch_repeat(noise, model_pred.shape[0] // noise.shape[0])
        
        if self.config.prediction_type == "target":
            # In target mode, the discriminator predicts directly from the unet output
            discriminator_input = model_pred
        else:
            predicted_latents = get_predicted_latents(noisy_latents, model_pred, timesteps, noise_scheduler)

            if self.config.step_type == "relative":
                next_timesteps = torch.clamp(timesteps + self.config.step_offset, min=0, max=noise_scheduler.config.num_train_timesteps-1)
            elif self.config.step_type == "absolute":
                next_timesteps = torch.full_like(timesteps, self.config.step_offset)
            else:
                raise ValueError(f"Unknown step type {self.config.step_type}")

            if self.config.prediction_type == "step":
                # In step mode, the discriminator gets the simulated result of stepping the denoising process several steps.
                # We adjust the noise distribution based on the formula for P(q=x|q+r=z) where q,r are the noise distributions 
                # for [0, next_timesteps) and [next_timesteps, timesteps) respectively, and z is the already-drawn total noise.
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                betas_cumprod = 1 - alphas_cumprod
                betas_ratio = betas_cumprod[next_timesteps] / betas_cumprod[timesteps]
                betas_ratio = unsqueeze_like(betas_ratio, noisy_latents)
                adjusted_noise = ((1 - betas_ratio) ** 0.5) * torch.randn_like(predicted_latents) + (betas_ratio ** 0.5) * noise
                discriminator_input = noise_scheduler.add_noise(predicted_latents, adjusted_noise, next_timesteps)
            elif self.config.prediction_type == "noisy_step":
                # In noisy step mode, we take the predicted denoised latents and add new noise
                # This helps regularize the discriminator. See https://arxiv.org/abs/2206.02262
                discriminator_input = noise_scheduler.add_noise(predicted_latents, torch.randn_like(predicted_latents), next_timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.config.prediction_type}")
            
        if self.config.in_channels > discriminator_input.shape[1]:
            # Some discriminator modes get both the unet input and output
            discriminator_input = torch.cat((noisy_latents, discriminator_input), 1)

        return discriminator_input
        
