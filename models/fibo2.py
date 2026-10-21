import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

import comfy.latent_formats
from comfy.ldm.fibo2.model import gist_count, pos_ids_x

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader

# ComfyUI's sampling_settings for Fibo 2.0 use shift=1.15 as mu; diffusers' scheduler wants exp(mu).
DEFAULT_SHIFT = math.exp(1.15)

# utils/patches.py replaces comfy.ldm.fibo2.model.modulate/apply_gate with autograd-safe (out-of-place)
# versions. The upstream ones write the timestep-zero slices into x in place, which autograd rejects.


def load_image_for_text_encoder(path):
    pil_img = Image.open(path)
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')
    # ComfyUI's text encoders take images as (batch, h, w, c) float tensors in [0, 1].
    return torch.from_numpy(np.array(pil_img)).float().div_(255.0).unsqueeze(0)


class Fibo2Pipeline(ComfyPipeline):
    name = 'fibo2'
    checkpointable_layers = ['NoiseRefinerLayer', 'ContextLayer', 'TransformerLayer']
    adapter_target_modules = ['Block', 'PerceiverLayer']
    # Matched against the relative module.child parameter names that ComfyPipeline.dequantize sees.
    # The small projections/modulations stay in dtype while the big GEMMs get cast to diffusion_model_dtype.
    keep_in_high_precision = [
        'x_embedder', 't_embedder',
        '0.weight', '1.weight', '2.weight',  # t_embedder MLP and the adaLN modulation projections
        'linear.weight',  # final_layer
        'latent_queries',  # Perceiver queries
    ]
    spatial_compression = 16
    channels = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional cap on the cached text length. None means each caption is cached at its natural length.
        self.fixed_text_length = int(self.model_config.get('text_length', 0)) or None

        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.latent_format = comfy.latent_formats.Flux2()
        self.sample_shift = DEFAULT_SHIFT
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=self.sample_shift)
        sigmas = torch.linspace(1.0, 1 / self.sample_steps, self.sample_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device='cuda')

    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        # The 3rd parameter makes DatasetManager pass the dataset's control_file list, so an edit
        # dataset conditions the text encoder on the reference image like the ComfyUI node does.
        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_files=None):
            embeds = []
            for i, caption in enumerate(captions):
                images = []
                if control_files is not None and control_files[i] is not None:
                    images = [load_image_for_text_encoder(control_files[i])]
                tokens = text_encoder.tokenize(caption, images=images)
                o = text_encoder.encode_from_tokens_scheduled(tokens)
                # (1, seq, 46080) -> (seq, 46080). The length depends on the caption and the reference
                # image, so embeddings stay variable length in the cache and are padded per batch.
                text_embeds = o[0][0][0].to(self.dtype)
                if self.fixed_text_length is not None:
                    text_embeds = text_embeds[:self.fixed_text_length]
                embeds.append(text_embeds)
            return {f'text_embeds_{te_idx}': embeds}

        return fn

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        if torch.is_tensor(text_embeds):
            text_embeds = list(text_embeds)
        # Pad to the longest caption in the batch. Doing this here (instead of at caching time) keeps the
        # gist token count based on the real text length instead of a fixed padded length.
        max_len = max(e.shape[0] for e in text_embeds)
        text_embeds = torch.stack([
            torch.cat((e, e.new_zeros(max_len - e.shape[0], e.shape[1]))) for e in text_embeds
        ])
        return (text_embeds,)

    def get_call_vae_fn(self, vae):
        # Same as the parent, but supports a reference/control image (edit datasets).
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent_vae_fn(args[0])
            elif len(args) == 2:
                tensor, control_tensor = args
                result = parent_vae_fn(tensor)
                control_result = parent_vae_fn(control_tensor)
                result['control_latents'] = control_result['latents']
                return result
            else:
                raise RuntimeError(f'Unexpected number of args: {len(args)}')
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        conds = self.get_conds(inputs)

        bs, c, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            # Fibo 2.0's latents are already 128-channel at 16x (ComfyUI/Flux2 layout) and the DiT has no
            # patch embedding, so every latent pixel is a token: image_seq_len = h * w. The inherited
            # Flux formula divides by 2 for its 2x2 patchify, which would be 4x too small here.
            # At 1024x1024 this gives 4096 tokens and mu = 1.15, matching the official pipeline.
            mu = get_lin_function(y1=0.5, y2=1.15)(h * w)
            t = time_shift(mu, 1.0, t)
        else:
            # The official schedule is always shifted.
            t = (t * DEFAULT_SHIFT) / (1 + (DEFAULT_SHIFT - 1) * t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        # Reference latents for image pair training. vae_encode already applied latent_format.process_in.
        # They are clean: the model treats everything outside the image span as t = 0.
        extra_inputs = ()
        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            assert control_latents.shape == latents.shape, (
                f"Control latents shape {control_latents.shape} doesn't match latents shape {latents.shape}"
            )
            extra_inputs = (control_latents,)

        return (noisy_latents, t, *conds) + extra_inputs, (target, mask)

    def to_layers(self):
        model = self.diffusion_model
        layers = [InitialLayer(model)]
        block_idx = 0
        for block in model.noise_refiner:
            layers.append(NoiseRefinerLayer(block, block_idx, self.offloader))
            block_idx += 1
        layers.append(ContextLayer(model))
        injection_layers = list(model.injection_layers)
        num_bundles = 1 + len(injection_layers)
        for i, block in enumerate(model.layers):
            bundle_idx = injection_layers.index(i) + 1 if i in injection_layers else None
            layers.append(TransformerLayer(block, block_idx, self.offloader, bundle_idx, num_bundles, model.max_gist_tokens))
            block_idx += 1
        layers.append(FinalLayer(model))
        return layers

    def _ordered_blocks(self):
        model = self.diffusion_model
        return list(model.noise_refiner) + list(model.layers)

    def enable_block_swap(self, blocks_to_swap):
        model = self.diffusion_model
        noise_refiner = model.noise_refiner
        blocks = model.layers
        num_blocks = len(noise_refiner) + len(blocks)

        assert blocks_to_swap <= num_blocks - 2, (
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks.'
        )

        self.offloader = ModelOffloader(
            'Fibo2Block', self._ordered_blocks(), num_blocks, blocks_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing']
        )
        model.noise_refiner = None
        model.layers = None
        model.to('cuda')
        model.noise_refiner = noise_refiner
        model.layers = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Keep the underlying model around for helpers instead of registering it a second time.
        self.model = [model]
        self.x_embedder = model.x_embedder
        self.t_embedder = model.t_embedder
        self.rope_embedder = model.rope_embedder
        self.max_gist_tokens = model.max_gist_tokens

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, timesteps, text_embeds, *extra = inputs
        device = x.device
        bs, _, h, w = x.shape

        num_gist = gist_count(text_embeds.shape[1], self.max_gist_tokens)
        plane = self.max_gist_tokens + 1

        tokens = [x.flatten(2).transpose(1, 2)]
        ids = [
            torch.zeros(1, num_gist, 3, dtype=torch.float32, device=device),
            pos_ids_x(plane, h, w, 1, device),
        ]
        ids[0][0, :, 0] = torch.arange(1, num_gist + 1, dtype=torch.float32, device=device)
        if len(extra) > 0:
            control = extra[0]
            tokens.append(control.flatten(2).transpose(1, 2))
            ids.append(pos_ids_x(plane + 1, control.shape[-2], control.shape[-1], 1, device))

        img = self.x_embedder(torch.cat(tokens, dim=1))
        # Replicate the rope positions to the batch dimension so that gradient accumulation can split
        # every model input along dim 0 (utils/dataset.py split_batch).
        freqs_cis = self.rope_embedder(torch.cat(ids, dim=1).expand(bs, -1, -1)).movedim(1, 2)

        # First half is t = 0 for the gist/reference tokens, second half modulates the noisy image.
        # Kept as two batch-sized tensors so gradient accumulation splits them into the right pieces.
        adaln_mixed = self.t_embedder(torch.cat((torch.zeros_like(timesteps), timesteps * 1000)), img.dtype)
        adaln_zero, adaln_noisy = adaln_mixed[:bs], adaln_mixed[bs:]

        hw = torch.tensor([h, w], dtype=torch.int64, device=device).repeat(bs, 1)
        return make_contiguous(img, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw)


class NoiseRefinerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw = inputs

        h, w = hw[0].tolist()
        n_img = h * w
        has_refs = img.shape[1] > n_img
        num_gist = freqs_cis.shape[1] - img.shape[1]

        # With references, the whole refiner runs on adaln_mixed and only the image span is noisy.
        if has_refs:
            adaln = torch.cat((adaln_zero, adaln_noisy), dim=0)
            index = [(0, n_img)]
        else:
            adaln = adaln_noisy
            index = None

        self.offloader.wait_for_block(self.block_idx)
        img = self.layer(img, freqs_cis[:, num_gist:], adaln, index)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw)


class ContextLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.perceiver = model.perceiver
        self.max_gist_tokens = model.max_gist_tokens
        self.num_bundles = 1 + len(model.injection_layers)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw = inputs

        num_gist = gist_count(text_embeds.shape[1], self.max_gist_tokens)
        # The Perceiver reads the first text bundle; the injection blocks read the other five.
        bundle = text_embeds.chunk(self.num_bundles, dim=-1)[0]
        gist = self.perceiver(bundle, num_gist)
        x = torch.cat((gist, img), dim=1)

        return make_contiguous(x, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader, bundle_idx, num_bundles, max_gist_tokens):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader
        self.bundle_idx = bundle_idx
        self.num_bundles = num_bundles
        self.max_gist_tokens = max_gist_tokens

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw = inputs

        h, w = hw[0].tolist()
        n_img = h * w
        num_gist = gist_count(text_embeds.shape[1], self.max_gist_tokens)
        image_index = [(num_gist, num_gist + n_img)]
        adaln_mixed = torch.cat((adaln_zero, adaln_noisy), dim=0)
        context = text_embeds.chunk(self.num_bundles, dim=-1)[self.bundle_idx] if self.bundle_idx is not None else None

        self.offloader.wait_for_block(self.block_idx)
        x = self.layer(x, freqs_cis, adaln_mixed, image_index, context)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.in_channels = model.final_layer.linear.out_features
        self.max_gist_tokens = model.max_gist_tokens

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, freqs_cis, text_embeds, adaln_zero, adaln_noisy, hw = inputs

        h, w = hw[0].tolist()
        n_img = h * w
        num_gist = gist_count(text_embeds.shape[1], self.max_gist_tokens)
        out = self.final_layer(x[:, num_gist:num_gist + n_img], adaln_noisy)
        return out.transpose(1, 2).reshape(x.shape[0], self.in_channels, h, w)
