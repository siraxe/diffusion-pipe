import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from diffusers import FlowMatchEulerDiscreteScheduler

import comfy.latent_formats
from comfy.ldm.flux.layers import ModulationOut, timestep_embedding

from models.base import ComfyPipeline, make_contiguous, maybe_pad
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader

# The blocks below call self.layer(x, spans, pe) directly. utils/patches.py replaces
# comfy.ldm.flux3.model.Block.forward with an autograd-safe (out-of-place) version.


class Flux3Pipeline(ComfyPipeline):
    name = 'flux3'
    checkpointable_layers = ['TxtBlockLayer', 'RefBlockLayer', 'ImageBlockLayer', 'SingleTransformerLayer']
    adapter_target_modules = ['Block']
    # Matched against the module.child parameter names that ComfyPipeline.dequantize sees. These small
    # projections/modulations are kept in high precision while the big GEMMs get cast to
    # diffusion_model_dtype.
    keep_in_high_precision = [
        'image.weight', 'image_cond.weight',  # emb_in
        'txt_in',
        'in_layer.weight', 'out_layer.weight',  # time_in
        'lin.weight',  # early/single stream modulations
        'linear.weight', '1.weight',  # final_layer
    ]
    spatial_compression = 16
    channels = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Flux3 pads text to a multiple of 80 tokens. We always pad/truncate to a fixed length so that
        # cached text embeddings all have the same shape and can be batched together.
        self.fixed_text_length = max(80, -(-int(self.model_config.get('text_length', 640)) // 80) * 80)

        with safetensors.safe_open(self.model_config['diffusion_model'], framework='pt', device='cpu') as f:
            if any('ref_time_in.in_layer.weight' in key for key in f.keys()):
                raise NotImplementedError(
                    'This looks like a Flux3 flow-map (step-distilled) checkpoint, which is not supported for '
                    'training. Please use the base Flux3 model.'
                )

        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.latent_format = comfy.latent_formats.Flux2()
        # ComfyUI's sampling settings use shift=1.93586 == ln(6.93); the diffusers scheduler wants the raw shift.
        self.sample_shift = 6.93
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

        fixed_length = self.fixed_text_length

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool]):
            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)
            original_min_length = tokenizer.min_length

            # Pad to the configured length (or the longest caption in the batch, whichever is longer).
            # This keeps the tokenizer's multiple-of-80 padding intact, and captions longer than the
            # configured length get truncated when we slice the embeddings below.
            min_length = fixed_length
            for text in captions:
                tokens = text_encoder.tokenize(text)
                if 'keep_vision' in tokens:
                    del tokens['keep_vision']
                for v in tokens.values():
                    min_length = max(min_length, len(v[0]))

            tokenizer.min_length = min_length
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = text_encoder.tokenize(text)
                for k, v in tokens.items():
                    if k == 'keep_vision':
                        continue
                    token_list = v[0]
                    maybe_pad(token_list, min_length, tokenizer)
                    tokens_dict[k].append(token_list)

            if 'keep_vision' in tokens:
                tokens_dict['keep_vision'] = tokens['keep_vision']

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0].to(self.dtype)[:, :fixed_length]
            extra = o[0][1]
            if 'attention_mask' in extra:
                attention_mask = extra['attention_mask'][:, :fixed_length]
            else:
                attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device)

            tokenizer.min_length = original_min_length
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_mask,
            }

        return fn

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        if torch.is_tensor(text_embeds):
            text_embeds = text_embeds[:, :self.fixed_text_length]
        else:
            text_embeds = torch.stack([u[:self.fixed_text_length, :] for u in text_embeds])
        return (text_embeds,)

    def get_call_vae_fn(self, vae):
        # Same as the parent, but supports a reference/control image (edit datasets)
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

    def to_layers(self):
        model = self.diffusion_model
        layers = [InitialLayer(model)]
        block_idx = 0
        for block in model.txt_mode_blocks:
            layers.append(TxtBlockLayer(block, block_idx, self.offloader))
            block_idx += 1
        for block in model.content_mode_blocks['image_cond']:
            layers.append(RefBlockLayer(block, block_idx, self.offloader))
            block_idx += 1
        for block in model.content_mode_blocks['image']:
            layers.append(ImageBlockLayer(block, block_idx, self.offloader))
            block_idx += 1
        layers.append(CombineSequences(model))
        for block in model.single_blocks:
            layers.append(SingleTransformerLayer(block, block_idx, self.offloader))
            block_idx += 1
        layers.append(FinalLayer(model))
        return layers

    def _ordered_blocks(self):
        model = self.diffusion_model
        return (list(model.txt_mode_blocks) + list(model.content_mode_blocks['image_cond'])
                + list(model.content_mode_blocks['image']) + list(model.single_blocks))

    def enable_block_swap(self, blocks_to_swap):
        model = self.diffusion_model
        txt_blocks = model.txt_mode_blocks
        content_mode_blocks = model.content_mode_blocks
        single_blocks = model.single_blocks
        num_blocks = (len(txt_blocks) + len(content_mode_blocks['image_cond'])
                      + len(content_mode_blocks['image']) + len(single_blocks))

        assert blocks_to_swap <= num_blocks - 2, (
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks.'
        )

        self.offloader = ModelOffloader(
            'Flux3Block', self._ordered_blocks(), num_blocks, blocks_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing']
        )
        model.txt_mode_blocks = None
        model.content_mode_blocks = None
        model.single_blocks = None
        model.to('cuda')
        model.txt_mode_blocks = txt_blocks
        model.content_mode_blocks = content_mode_blocks
        model.single_blocks = single_blocks
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
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        # Reference/control latents for edit training. vae_encode already applied latent_format.process_in.
        extra_inputs = ()
        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            assert control_latents.shape == latents.shape, (
                f"Control latents shape {control_latents.shape} doesn't match latents shape {latents.shape}"
            )
            extra_inputs = (control_latents,)

        return (noisy_latents, t, *conds) + extra_inputs, (target, mask)


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.emb_in = model.emb_in
        self.txt_in = model.txt_in
        self.time_in = model.time_in
        self.early_stream_modulations = model.early_stream_modulations
        self.pe_embedder = model.pe_embedder
        self.image_rope_scale = model.image_rope_scale

    def _image_ids(self, h, w, index, device):
        # (t, row, col, l); references are told apart on the l axis
        ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * self.image_rope_scale
        ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * self.image_rope_scale
        ids[..., 3] = index
        return ids.flatten(0, 1)

    def _text_ids(self, length, device):
        ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
        ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
        return ids

    def _rope(self, ids):
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, timesteps, txt_embeds, *extra = inputs
        device = x.device
        bs, _, h, w = x.shape

        img = self.emb_in['image'](x.flatten(2).transpose(1, 2))
        if len(extra) > 0:
            control = extra[0]
            refs = self.emb_in['image_cond'](control.flatten(2).transpose(1, 2))
            ref_ids = self._image_ids(control.shape[-2], control.shape[-1], -1, device)
        else:
            refs = img.new_zeros((bs, 0, img.shape[-1]))
            ref_ids = torch.zeros((0, 4), device=device, dtype=torch.float32)
        txt = self.txt_in(txt_embeds)

        # the reference embeds t in the compute dtype; image_cond and text sit at t = 0
        vec = self.time_in(timestep_embedding(torch.cat([timesteps, timesteps.new_zeros(1)]).to(img.dtype), 256))
        vec_img, vec_zero = vec[:-1], vec[-1:]

        # per sample the joint sequence is [image, image_cond, text]
        pe = self._rope(torch.cat([self._image_ids(h, w, 0, device), ref_ids, self._text_ids(txt.shape[1], device)], dim=0))

        e_img = self.early_stream_modulations['image'](vec_img)[0]
        e_txt = self.early_stream_modulations['txt'](vec_zero)[0]
        e_ref = self.early_stream_modulations['image_cond'](vec_zero)[0]

        img_len = torch.tensor(img.shape[1], dtype=torch.int64, device=device)
        ref_len = torch.tensor(refs.shape[1], dtype=torch.int64, device=device)
        txt_len = torch.tensor(txt.shape[1], dtype=torch.int64, device=device)
        hw = torch.tensor([h, w], dtype=torch.int64, device=device)

        return make_contiguous(
            img, txt, refs, pe, vec_img, vec_zero,
            e_img.shift, e_img.scale, e_img.gate,
            e_txt.shift, e_txt.scale, e_txt.gate,
            e_ref.shift, e_ref.scale, e_ref.gate,
            img_len, ref_len, txt_len, hw,
        )


class TxtBlockLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw = inputs

        self.offloader.wait_for_block(self.block_idx)
        mod = ModulationOut(em[3], em[4], em[5])
        start = img_len + ref_len
        txt = self.layer(txt, [(0, txt.shape[1], mod)], pe[:, start:start + txt.shape[1]])
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw)


class RefBlockLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw = inputs

        # Always participate in the block swap scheduling, even when there are no references
        self.offloader.wait_for_block(self.block_idx)
        if refs.shape[1] > 0:
            mod = ModulationOut(em[6], em[7], em[8])
            refs = self.layer(refs, [(0, refs.shape[1], mod)], pe[:, img_len:img_len + refs.shape[1]])
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw)


class ImageBlockLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw = inputs

        self.offloader.wait_for_block(self.block_idx)
        mod = ModulationOut(em[0], em[1], em[2])
        img = self.layer(img, [(0, img.shape[1], mod)], pe[:, :img.shape[1]])
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw)


class CombineSequences(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.single_stream_modulations = model.single_stream_modulations

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, refs, pe, vec_img, vec_zero, *em, img_len, ref_len, txt_len, hw = inputs

        x = torch.cat([img, refs, txt], dim=1)
        m_img = self.single_stream_modulations['image'](vec_img)[0]
        m_ref = self.single_stream_modulations['image_cond'](vec_zero)[0]
        m_txt = self.single_stream_modulations['txt'](vec_zero)[0]

        return make_contiguous(
            x, pe, vec_img,
            m_img.shift, m_img.scale, m_img.gate,
            m_ref.shift, m_ref.scale, m_ref.gate,
            m_txt.shift, m_txt.scale, m_txt.gate,
            img_len, ref_len, txt_len, hw,
        )


class SingleTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, pe, vec_img, s_s, s_sc, s_g, r_s, r_sc, r_g, t_s, t_sc, t_g, img_len, ref_len, txt_len, hw = inputs

        self.offloader.wait_for_block(self.block_idx)
        ref_end = img_len + ref_len
        # when there are no references the middle span is empty, which is a no-op
        spans = [
            (0, img_len, ModulationOut(s_s, s_sc, s_g)),
            (img_len, ref_end, ModulationOut(r_s, r_sc, r_g)),
            (ref_end, ref_end + txt_len, ModulationOut(t_s, t_sc, t_g)),
        ]
        x = self.layer(x, spans, pe)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, pe, vec_img, s_s, s_sc, s_g, r_s, r_sc, r_g, t_s, t_sc, t_g, img_len, ref_len, txt_len, hw)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.in_channels = model.in_channels

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, pe, vec_img, *_mods, img_len, ref_len, txt_len, hw = inputs

        out = self.final_layer['image'](x[:, :img_len], vec_img)
        h, w = hw.tolist()
        return out.transpose(1, 2).reshape(x.shape[0], self.in_channels, h, w)
