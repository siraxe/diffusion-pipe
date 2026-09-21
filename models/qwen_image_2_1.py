import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from PIL import Image

from models.base import ComfyPipeline, PreprocessMediaFile, make_contiguous, convert_crop_and_resize
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.latent_formats
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.qwen_image21.model import _split_rows

# Default guidance scale for Qwen-Image 2.1 (used for both training CFG and sampling).
# Override via model_config: cfg= (training) or guidance_scale= (sampling).
DEFAULT_GUIDANCE_SCALE = 3.0

# Monkey-patch _gated_residual to avoid in-place operation on views of leaf variables
# (which breaks autograd during training). The original uses in-place addcmul_ on views.
import comfy.ldm.qwen_image21.model as _qwen_model

_original_gated_residual = _qwen_model._gated_residual

def _safe_gated_residual(x, y, gate, prefix_len):
    """Clone x before in-place ops to avoid 'view of leaf Variable' error during training."""
    g_prefix, g_target = gate
    x = x.clone()
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x

_qwen_model._gated_residual = _safe_gated_residual


class QwenImage21Pipeline(ComfyPipeline):
    name = 'qwen_image_2_1'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    adapter_target_modules = ['QwenImage21TransformerBlock']
    keep_in_high_precision = ['txt_in', 'img_in', 'time_text_embed', 'modulation', 'norm_out', 'proj_out']
    spatial_compression = 16
    channels = 64
    # Reference images are seen by the Qwen3-VL text encoder at 32 px per vision token and by the
    # VAE at 16 px per latent token (4 latent tokens per vision token), so everything needs to line
    # up on a 32 px grid.
    pixels_round_to_multiple = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.QwenImage21()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        self.cfg = self.model_config.get('cfg', DEFAULT_GUIDANCE_SCALE)
        if self.cfg > 1:
            # Because uncond branch is no_grad() but tensors passed between pipeline parallel layers must require grad.
            assert self.config['pipeline_stages'] == 1, 'CFG training requires pipeline_stages=1'

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def vae_decode(self, latents):
        # The VAE outputs RGBA. Drop the alpha channel.
        img = super().vae_decode(latents)
        return img[..., :3]

    def vae_encode(self, img):
        # The VAE expects RGBA input. The dataset loader provides RGB, so pad an
        # opaque alpha channel. Works for both images (b c h w) and video
        # (b c f h w).
        if img.shape[1] == 3:
            img = torch.cat([img, torch.ones_like(img[:, :1])], dim=1)
        return super().vae_encode(img)

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        return text_embeds, attention_mask

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=None):
        """Override to default guidance_scale to 3.0, matching ai-toolkit's behavior.

        Qwen-Image 2.1 is trained with CFG and benefits from a guidance scale of
        ~3.0 for best quality. The base class defaults to cfg=1 (no guidance),
        which produces washed-out results.
        """
        if cfg is None:
            cfg = self.model_config.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)
        super().prepare_sample_test(prompt, negative_prompt, cfg)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _load_control_image(self, path, size_bucket):
        # Resize a reference image to exactly the bucket the VAE-cached control latents were encoded
        # at, so the vision token grid (32 px) and the latent token grid (16 px) agree.
        if size_bucket is None:
            raise RuntimeError(
                'Qwen-Image 2.1 edit training requires size buckets so the text encoder and the VAE '
                'see reference images at the same resolution. Size buckets should be enabled by '
                'setting a resolution on the dataset.'
            )
        width, height, _ = size_bucket
        pil_img = Image.open(path)
        pil_img = convert_crop_and_resize(pil_img, (width, height))
        # (1, H, W, 3) in [0, 1], which is the format the ComfyUI tokenizer expects.
        return torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).movedim(1, -1)

    def get_call_text_encoder_fn(self, text_encoder):
        generic_fn = super().get_call_text_encoder_fn(text_encoder)
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions, is_video, control_file=None, size_bucket=None):
            assert not any(is_video)
            if control_file is None:
                return generic_fn(captions, is_video)

            # Edit dataset: encode one caption at a time so the image slot bookkeeping (which is
            # where each reference image's latents go in the text sequence) stays per-sample.
            embeds, slot_lists = [], []
            for caption, c_path, bucket in zip(captions, control_file, size_bucket):
                image = self._load_control_image(c_path, bucket).to('cuda')
                # The tokenizer builds the "<image1><|vision_start|><|image_pad|><|vision_end|>"
                # template and the text encoder records the slot positions in extra['image_slots'].
                tokens = text_encoder.tokenize(caption, images=[image])
                o = text_encoder.encode_from_tokens_scheduled(tokens)
                cond, extra = o[0][0], o[0][1]
                embeds.append(cond[0])
                slot_lists.append(extra.get('image_slots', []))

            max_seq_len = max([e.size(0) for e in embeds])
            batch_embeds = torch.stack(
                [torch.cat([e, e.new_zeros(max_seq_len - e.size(0), e.size(1))]) for e in embeds]
            ).to(self.dtype)
            attention_mask = torch.stack(
                [torch.cat([torch.ones(e.size(0), dtype=torch.int64, device=e.device),
                            torch.zeros(max_seq_len - e.size(0), dtype=torch.int64, device=e.device)]) for e in embeds]
            )
            image_slots = torch.tensor(slot_lists, dtype=torch.int64, device=batch_embeds.device)
            return {
                f'text_embeds_{te_idx}': batch_embeds,
                f'attention_mask_{te_idx}': attention_mask,
                'image_slots': image_slots,
            }

        return fn

    def get_call_vae_fn(self, vae):
        def fn(images, control_images=None):
            latents = self.vae_encode(images)
            result = {'latents': latents}
            if control_images is not None:
                # Reference image latents. These are spliced into the token sequence by the
                # transformer at the positions the text encoder reserved for them.
                result['control_latents'] = self.vae_encode(control_images)
            return result
        return fn

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        if latents.dim() == 5:
            assert latents.shape[2] == 1
            latents = latents.squeeze(2)
        prompt_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        mask = inputs['mask']

        # prompt embeds are variable length
        max_seq_len = max([e.size(0) for e in prompt_embeds])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )

        # Trim off the padding, which is only needed to form the batch.
        max_text_len = prompt_embeds_mask.sum(dim=1).max().item()
        prompt_embeds = prompt_embeds[:, :max_text_len, :]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_text_len].to(torch.bool)

        bs, c, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask.float(), size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

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

        # The Qwen-Image 2.1 scheduler shift: 0.5 at 256 image tokens, 0.9 at 8192 (0.69 at
        # 1024x1024). The latents are not patched, so an image token is one latent position.
        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)
        elif not self.model_config.get('disable_shift', False):
            mu = get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)(h * w)
            t = time_shift(mu, 1.0, t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # Reference images for editing. The whole batch must agree on the slot layout, so if any
        # sample's embeddings were encoded without references (e.g. caption dropout), the references
        # are dropped for the entire batch.
        extra = tuple()
        image_slots = inputs.get('image_slots', None)
        if image_slots is not None:
            if torch.is_tensor(image_slots):
                image_slots = list(image_slots)
            if 'control_latents' in inputs and all(int(s.numel()) > 0 for s in image_slots):
                control_latents = inputs['control_latents'].float()
                if control_latents.dim() == 5:
                    assert control_latents.shape[2] == 1
                    control_latents = control_latents.squeeze(2)
                assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
                slots = image_slots[0]
                assert all(torch.equal(s, slots) for s in image_slots), 'image slots must be the same across the batch'
                extra = (control_latents, slots)

        # CFG training: include unconditional embeddings if cfg > 1
        if self.cfg > 1:
            uncond_embeds = self.uncond_dict['text_embeds_0'].unsqueeze(0).repeat(bs, 1, 1)
            uncond_mask = self.uncond_dict['attention_mask_0'].unsqueeze(0).repeat(bs, 1).to(torch.bool)
            # Trim uncond to matching length
            max_uncond_len = uncond_mask.sum(dim=1).max().item()
            uncond_embeds = uncond_embeds[:, :max_uncond_len, :]
            uncond_mask = uncond_mask[:, :max_uncond_len]
            return (x_t, t, prompt_embeds, prompt_embeds_mask, uncond_embeds, uncond_mask) + extra, (target, mask)
        else:
            return (x_t, t, prompt_embeds, prompt_embeds_mask) + extra, (target, mask)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model, cfg=self.cfg))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.transformer_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.transformer_blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


def make_attn_fn(segments):
    # Mirrors comfy.ldm.qwen_image21.model.block_causal_attention, minus the sampling-only KV cache:
    # one attention call per sequence segment (causal for text, full-attention-to-prefix for images).
    def attn(q, k, v, heads):
        # q, k, v: (B, N, H, D)
        outs = [
            optimized_attention(q[:, start:end].flatten(2), k[:, :end].flatten(2), v[:, :end].flatten(2), heads, mask=mask)
            for start, end, mask in segments
        ]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return attn


def _fold_key_valid(segments, key_valid):
    # Padded caption positions must never be attended to. Fold the batched key-validity mask into
    # the per-segment masks: text segments keep their causal structure, image segments (which attend
    # to everything before them) become key-validity masks. Masks are 3D (B, n_queries, n_keys) or
    # (B, 1, n_keys), which the attention backends broadcast over heads.
    new_segments = []
    for start, end, seg_mask in segments:
        if seg_mask is None:
            mask = key_valid[:, :end].unsqueeze(1)
        else:
            mask = seg_mask.unsqueeze(0) & key_valid[:, None, :end]
        if bool(mask.all()):
            # A fully-True mask costs more than it saves with some attention backends.
            mask = None
        new_segments.append((start, end, mask))
    return new_segments


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pe_embedder = model.pe_embedder
        self.time_text_embed = model.time_text_embed
        self.txt_in = model.txt_in
        self.img_in = model.img_in
        self.modulation = model.modulation
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, t, context, context_mask, *extra = inputs
        model = self.model[0]
        B, _, H, W = x.shape

        # Detect unconditional branch (CFG training)
        has_uncond = len(extra) >= 2 and torch.is_tensor(extra[0]) and extra[0].dim() == 3 and extra[0].shape[0] == B
        if has_uncond:
            context_uncond, context_mask_uncond = extra[0], extra[1]
            extra = extra[2:]
        else:
            context_uncond, context_mask_uncond = None, None

        ref_latents, image_slots = [], []
        if len(extra) == 2:
            ref_latents = [extra[0]]
            image_slots = [int(s) for s in extra[1].tolist()]

        # Build the joint sequence: text with reference latents spliced in at their slots, target
        # image last. Reuses the ComfyUI implementation so the key layout stays identical.
        hidden_states, pe, segments = model.build_sequence(x, context, ref_latents, image_slots)
        prefix_len = hidden_states.shape[1] - H * W

        if not bool(context_mask.all()):
            # Joint positions of the text tokens, mirroring the layout walk in build_sequence.
            L = context.shape[1]
            if ref_latents:
                slots = (image_slots + [L] * len(ref_latents))[:len(ref_latents)]
                bounds = [0] + slots + [L]
            else:
                bounds = [0, L]
            text_positions, pos = [], 0
            for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
                n = end - start
                if n > 0:
                    text_positions.extend(range(pos, pos + n))
                    pos += n
                pos += max(img.shape[-2], img.shape[-3])
            key_valid = torch.ones(B, hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            key_valid[:, torch.tensor(text_positions, device=hidden_states.device)] = context_mask.bool()
            segments = _fold_key_valid(segments, key_valid)

        # The pipeline rounds t*1000 and t to the compute dtype; text and reference tokens modulate
        # from t = 0 (the trailing row), target image tokens from the sampled timestep.
        dtype = hidden_states.dtype
        t = ((t * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)], dim=0), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)

        # pe must be the same dtype as hidden_states, or training hangs but only when
        # pipeline_stages>=3. (???)
        pe = pe.to(device=hidden_states.device, dtype=hidden_states.dtype)
        pe.requires_grad_(True)

        # Serialize the segment layout so the layer tuple only contains tensors (needed for pipeline
        # parallelism): [num_segments, prefix_len, H, W, start, end, has_mask, ...]. The masks of the
        # segments that have one are appended in order after the core tensors.
        seg_masks = [m for (_, _, m) in segments if m is not None]
        meta = [len(segments), prefix_len, H, W]
        for start, end, m in segments:
            meta.extend([start, end, 1 if m is not None else 0])
        meta = torch.tensor(meta, dtype=torch.int64, device=hidden_states.device)

        # Conditional branch outputs
        outputs = make_contiguous(hidden_states, pe, scale1, gate1, scale2, gate2, temb, meta) + tuple(seg_masks)

        # Unconditional branch (CFG training) - run in no_grad to save memory
        if has_uncond:
            with torch.no_grad():
                hidden_states_uncond, pe_uncond, segments_uncond = model.build_sequence(
                    x, context_uncond, ref_latents, image_slots
                )
                # Build key_valid for uncond branch if needed
                if not bool(context_mask_uncond.all()):
                    L_uncond = context_uncond.shape[1]
                    if ref_latents:
                        slots_uncond = (image_slots + [L_uncond] * len(ref_latents))[:len(ref_latents)]
                        bounds_uncond = [0] + slots_uncond + [L_uncond]
                    else:
                        bounds_uncond = [0, L_uncond]
                    text_positions_uncond, pos_uncond = [], 0
                    for (start, end), img in zip(zip(bounds_uncond[:-1], bounds_uncond[1:]), ref_latents + [x]):
                        n = end - start
                        if n > 0:
                            text_positions_uncond.extend(range(pos_uncond, pos_uncond + n))
                            pos_uncond += n
                        pos_uncond += max(img.shape[-2], img.shape[-3])
                    key_valid_uncond = torch.ones(B, hidden_states_uncond.shape[1], dtype=torch.bool, device=hidden_states_uncond.device)
                    key_valid_uncond[:, torch.tensor(text_positions_uncond, device=hidden_states_uncond.device)] = context_mask_uncond.bool()
                    segments_uncond = _fold_key_valid(segments_uncond, key_valid_uncond)

                pe_uncond = pe_uncond.to(device=hidden_states_uncond.device, dtype=hidden_states_uncond.dtype)
                seg_masks_uncond = [m for (_, _, m) in segments_uncond if m is not None]
                prefix_len_uncond = hidden_states_uncond.shape[1] - H * W
                meta_uncond = [len(segments_uncond), prefix_len_uncond, H, W]
                for start, end, m in segments_uncond:
                    meta_uncond.extend([start, end, 1 if m is not None else 0])
                meta_uncond = torch.tensor(meta_uncond, dtype=torch.int64, device=hidden_states_uncond.device)

            uncond_tensors = (hidden_states_uncond, pe_uncond, meta_uncond) + tuple(seg_masks_uncond)
            for x in uncond_tensors:
                x.no_backward = True
            outputs = (*outputs, *uncond_tensors)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, pe, scale1, gate1, scale2, gate2, temb, meta, *seg_masks = inputs
        self.offloader.wait_for_block(self.block_idx)

        meta_list = meta.tolist()
        num_segments = meta_list[0]
        prefix_len = meta_list[1]
        segments, mask_idx = [], 0
        for i in range(num_segments):
            base = 4 + 3 * i
            start, end, has_mask = meta_list[base], meta_list[base + 1], meta_list[base + 2]
            if has_mask:
                segments.append((start, end, seg_masks[mask_idx]))
                mask_idx += 1
            else:
                segments.append((start, end, None))

        attn_fn = make_attn_fn(segments)
        # One shared modulation for every block: [scale1, gate1, scale2, gate2]. The prefix (t = 0)
        # row modulates text and reference tokens, the sampled-t rows the target image tokens.
        mod = (
            _split_rows(scale1),
            _split_rows(gate1.tanh()),
            _split_rows(scale2),
            _split_rows(gate2.tanh()),
            torch.zeros_like(scale1[:1, None]),
        )
        hidden_states = self.block(hidden_states, mod, pe, attn_fn, prefix_len)

        # Process unconditional branch if present (CFG training)
        # Uncond tensors are appended after conditional seg_masks:
        # hidden_states_uncond, pe_uncond, meta_uncond, *seg_masks_uncond
        if len(seg_masks) > mask_idx:
            # There are extra tensors for the unconditional branch
            uncond_start = mask_idx
            hidden_states_uncond = seg_masks[uncond_start]
            pe_uncond = seg_masks[uncond_start + 1]
            meta_uncond = seg_masks[uncond_start + 2]
            seg_masks_uncond = seg_masks[uncond_start + 3:]

            if hidden_states_uncond is None:
                # Backward pass recomputation: the unsloth checkpointer doesn't save
                # no_backward tensors, so they arrive as None. We don't need the uncond
                # branch for gradients, but must return the same number of values.
                uncond_tensors = (None,) * (3 + len(seg_masks_uncond))
            else:
                with torch.no_grad():
                    meta_list_uncond = meta_uncond.tolist()
                    num_segments_uncond = meta_list_uncond[0]
                    segments_uncond, mask_idx_uncond = [], 0
                    for i in range(num_segments_uncond):
                        base = 4 + 3 * i
                        start, end, has_mask = meta_list_uncond[base], meta_list_uncond[base + 1], meta_list_uncond[base + 2]
                        if has_mask:
                            segments_uncond.append((start, end, seg_masks_uncond[mask_idx_uncond]))
                            mask_idx_uncond += 1
                        else:
                            segments_uncond.append((start, end, None))

                    attn_fn_uncond = make_attn_fn(segments_uncond)
                    prefix_len_uncond = meta_list_uncond[1]
                    hidden_states_uncond = self.block(hidden_states_uncond, mod, pe_uncond, attn_fn_uncond, prefix_len_uncond)

                uncond_tensors = (hidden_states_uncond, pe_uncond, meta_uncond) + tuple(seg_masks_uncond)
                for x in uncond_tensors:
                    x.no_backward = True
            seg_masks = seg_masks[:mask_idx] + list(uncond_tensors)
        else:
            seg_masks = seg_masks[:mask_idx]

        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, pe, scale1, gate1, scale2, gate2, temb, meta) + tuple(seg_masks)


class FinalLayer(nn.Module):
    def __init__(self, model, cfg=1.0):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]
        self.cfg = cfg

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        hidden_states, pe, scale1, gate1, scale2, gate2, temb, meta, *seg_masks = inputs
        meta_list = meta.tolist()
        num_segments = meta_list[0]
        prefix_len, H, W = meta_list[1], meta_list[2], meta_list[3]
        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        output = self.proj_out(hidden_states)
        B = output.shape[0]
        output = output.transpose(1, 2).reshape(B, self.out_channels, H, W)

        # CFG training: combine conditional and unconditional predictions
        # Formula: out = (cond + (cfg-1) * uncond) / cfg  (rearranged CFG)
        # Count how many conditional seg_masks there are
        num_cond_masks = sum(1 for i in range(num_segments) if meta_list[4 + 3*i + 2] == 1)
        if self.cfg > 1 and len(seg_masks) > num_cond_masks:
            # Uncond tensors are appended after conditional seg_masks:
            # hidden_states_uncond, pe_uncond, meta_uncond, *seg_masks_uncond
            hidden_states_uncond = seg_masks[num_cond_masks]
            meta_uncond = seg_masks[num_cond_masks + 2]
            meta_list_uncond = meta_uncond.tolist()
            prefix_len_uncond, H_uncond, W_uncond = meta_list_uncond[1], meta_list_uncond[2], meta_list_uncond[3]
            with torch.no_grad():
                # Extract just the target image portion (skip prefix)
                hidden_states_uncond = self.norm_out(hidden_states_uncond[:, prefix_len_uncond:], temb[:-1])
                output_uncond = self.proj_out(hidden_states_uncond)
                output_uncond = output_uncond.transpose(1, 2).reshape(B, self.out_channels, H_uncond, W_uncond)
            # "raw" velocity from guidance-distilled output and own uncond (rearranged CFG equation).
            # Must run OUTSIDE no_grad or the cond branch's graph is severed and the loss has no grad_fn.
            output = (output + (self.cfg - 1) * output_uncond) / self.cfg

        return output
