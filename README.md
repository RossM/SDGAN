# SDGAN
 SDGAN: Tuning Stable Diffusion with an adversarial network

This is a research project aimed at enhancing Stable Diffusion with GAN training for better perceptual detail. Currently, no pretrained model weights are available. Things are still in flux and the model may change in ways that are not backwards compatible with trained weights. Use at your own risk.

TODO
* ☐ Investigate using different samples for generator and discriminator training
* ☑ ~~Test the effect of adding self-attention to all layers~~ Failed due to CUDA running out of memory
* ☐ Test the effect of different parameter initializations
* ☐ Investigate using cross attention

BIBLIOGRAPHY (incomplete)
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
* [Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262)
* [Scaling up GANs for Text-to-Image Synthesis](https://arxiv.org/abs/2303.05511)
* [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
* [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804)
* [Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models](https://arxiv.org/abs/2211.17091)
