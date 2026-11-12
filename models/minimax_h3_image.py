import os
import sys

# The repo's namespace `utils` package must be imported before the ComfyUI path is added to
# sys.path, or ComfyUI's regular `utils` package would shadow it (PEP 420 regular beats namespace).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import utils.common  # noqa: F401  (no-op when running through train.py, which imports it first)

sys.path.insert(0, os.path.join(_repo_root, 'submodules', 'ComfyUI'))

from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps

# Importing models.minimax_h3 applies the shared patches: eager rms_rope Attention,
# batch-enabled AdalnProj / unpack_audio, backward-safe _mod_scale_shift / _mod_gate,
# and the attention-mask-enabled MiniMax Qwen3-VL text encoder.
from models import minimax_h3
from models.base import ComfyPipeline, ModelWrapper, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, round_down_to_multiple
from utils.offloading import ModelOffloader

import comfy.latent_formats
import comfy.ldm.minimax.model
import comfy.model_management
import comfy.sd
import comfy.utils
from comfy.ldm.minimax.model import (
    PackedLayout, SEG_TAG, VISUAL_COND_TIMESTEP, patchify_video, rope_rotation_table, unpatchify_video,
)

FRAMERATE = 24  # only relevant for video files; this pipeline trains single images

# The Comfy-Org MiniMax-H3-Image checkpoint is a 6-step no-CFG distilled model whose PDD output
# heads were distilled on this sigma grid. The grid is also the training and sampling schedule:
# each training step draws one of these sigmas and fits it through its own head.
DEFAULT_SIGMA_GRID = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
]
# short edge of the copy of a reference image shown to the Qwen3-VL text encoder
VISION_SHORT_EDGE = 1024
# picture label used for edit conditioning (no angle brackets, unlike the video archs)
PICTURE_LABEL = 'Picture %d: '


class PreprocessMediaFileMinimaxImage(PreprocessMediaFile):
    # support_video=True so images come out as (c, 1, h, w) like the video pipeline expects.
    def __init__(self, config):
        PreprocessMediaFile.__init__(self, config, support_video=True, support_audio=False,
                                     framerate=FRAMERATE, round_height=32, round_width=32)

    def align_frames(self, frames):
        return max(round_down_to_multiple(frames, 17), 1)


def _mod_scale_shift_rows(h, shift, scale, mod_row):
    # per-token mod rows: h (B, S, D), shift/scale (B, R, D), mod_row (B, S) ints into R
    dim = h.shape[-1]
    gather_idx = mod_row.unsqueeze(-1).expand(-1, -1, dim)
    s = scale.gather(1, gather_idx).to(h.dtype)
    sh = shift.gather(1, gather_idx).to(h.dtype)
    return h * (1.0 + s) + sh


def _mod_gate_rows(x, gate, other, mod_row):
    dim = x.shape[-1]
    g = gate.gather(1, mod_row.unsqueeze(-1).expand(-1, -1, dim)).to(x.dtype)
    return x + other * g


class DiTBlock(nn.Module):
    """Per-token mod-row variant of the H3 block.

    Instead of one shared segment list for the whole batch, every token carries its own AdaLN
    mod row. This is what allows reference-image conditioning (whose vision-token tag runs differ
    per prompt) to be batched together with plain text prompts.
    """

    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, gate_compress=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = comfy.ldm.minimax.model.Attention(hidden, heads, head_dim, qk_eps, gate_compress=gate_compress,
                                                      dtype=dtype, device=device, operations=operations)
        self.mlp = comfy.ldm.minimax.model.MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = comfy.ldm.minimax.model.AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                                            dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                                            device=device, operations=operations)

    def forward(self, x, t_emb, mod_row, rope_freqs, transformer_options={}, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift_rows(self.norm1(x), shift_msa, scale_msa, mod_row)
        x = _mod_gate_rows(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options,
                                                  attention_mask=attention_mask), mod_row)
        h = _mod_scale_shift_rows(self.norm2(x), shift_mlp, scale_mlp, mod_row)
        return _mod_gate_rows(x, gate_mlp, self.mlp(h), mod_row)


class FinalLayer(nn.Module):
    """PDD-aware, batch-capable version of comfy.ldm.minimax.model.FinalLayer.

    The checkpoint's output head bank stores pdd_steps stacked heads per stream; block 0 is a full
    head and later blocks are offsets from it. This class is only constructed by the ComfyUI model
    loader (which passes pdd_steps); the training forward pass is driven directly by the pipeline's
    FinalLayer wrapper below.
    """

    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None, pdd_steps=1,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.video_dim = video_dim
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = comfy.ldm.minimax.model.AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                                            dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                                            device=device, operations=operations)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_out = operations.Linear(hidden, video_dim * pdd_steps, bias=True, dtype=torch.float32, device=device)
        self.audio_out = None
        if audio_dim is not None:
            self.audio_out = operations.Linear(hidden, audio_dim * pdd_steps, bias=True, dtype=torch.float32, device=device)

    @property
    def pdd_steps(self):
        return self.video_out.weight.shape[0] // self.video_dim


# Patch these after importing models.minimax_h3 (which patches them with video variants that
# don't support the image checkpoint's PDD head bank or per-token mod rows).
comfy.ldm.minimax.model.DiTBlock = DiTBlock
comfy.ldm.minimax.model.FinalLayer = FinalLayer


class MinimaxH3ImagePipeline(minimax_h3.MinimaxH3Pipeline):
    """MiniMax-H3-Image: the H3 DiT without the audio stream, distilled to a 6-step no-CFG
    text-to-image / image-edit model (Comfy-Org repack).

    Packed sequence is [text | reference images | target image], one latent frame. Edit training
    is driven by the standard `control_path` dataset mechanism: each control image is VAE-encoded
    at the target's size and packed as a reference block (pinned near the cond timestep), and shown
    to the Qwen3-VL text encoder as a `Picture i:` vision block.

    Model config:
        diffusion_model: Comfy-Org MiniMax-H3-Image checkpoint (must carry its config metadata,
            which declares the PDD sigma grid).
        vae: the H3 image VAE (same H3 encoder as the video model, 2D decoder).
        text_encoders: the MiniMax H3 Qwen3-VL text encoder, same as the video model.
        sigma_grid: optional override of the training sigmas (defaults to the checkpoint's
            PDD grid). Only used for multi-head (PDD) checkpoints.
        cfg: not supported, the model is guidance-distilled (must be <= 1).
    """

    name = 'minimax_h3_image'
    keep_in_high_precision = ['time_embedder', 'condition_proj', 'final_layer', 'rope.inv_freq', 'token_refiner', 'video_patch_proj', 'adaln_t_table']

    def __init__(self, *args, **kwargs):
        ComfyPipeline.__init__(self, *args, **kwargs)
        self.is_video_vae = True
        self.latent_format = comfy.latent_formats.MiniMaxH3Video()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.framerate = FRAMERATE

        self.cfg = self.model_config.get('cfg', 1.0)
        assert self.cfg <= 1, 'MiniMax-H3-Image is guidance-distilled (no CFG); set cfg <= 1'

        # image VAE: same H3 encoder as the video model, no audio VAE
        import types

        def load_fn():
            sd = comfy.utils.load_torch_file(self.model_config['vae'])
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()

            def vae_encode_crop_pixels(self, pixels):
                if not self.crop_input:
                    return pixels

                downscale_ratio = self.spacial_compression_encode()

                dims = pixels.shape[-3:-1]
                for d in range(len(dims)):
                    x = (dims[d] // downscale_ratio) * downscale_ratio
                    x_offset = (dims[d] % downscale_ratio) // 2
                    if x != dims[d]:
                        pixels = pixels.narrow(d + 1, x_offset, x)
                return pixels

            # patch this to handle 5D tensor (original code expects 4D)
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, vae)
            return vae

        self.vae = ModelWrapper(load_fn)

    def _get_sigma_grid(self):
        """Full descending sigma grid (pdd_steps + 1 values ending at 0.0), or None for
        single-head checkpoints (which train with generic flow-matching timestep sampling)."""
        pdd_steps = self.diffusion_model.final_layer.pdd_steps
        if pdd_steps <= 1:
            return None
        sigma_grid = self.model_config.get('sigma_grid', None)
        if sigma_grid is None:
            return [float(s) for s in self.diffusion_model.pdd_sigmas]
        grid = [float(s) for s in sigma_grid] + [0.0]
        assert len(grid) - 1 == pdd_steps, f'sigma_grid must have {pdd_steps} values'
        return grid

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFileMinimaxImage(self.config)

    def vae_encode(self, img: torch.Tensor):
        # video model's vae_encode also encodes audio through a second VAE;
        # this pipeline has a single (video) VAE
        img = img.movedim(1, -1)
        latents = self.vae._model.encode(img)
        if self.latent_format is not None:
            # some older models do this in prepare_inputs() so it can be None
            latents = self.latent_format.process_in(latents)
        return latents

    def get_call_vae_fn(self, vae):
        def fn(images, control_tensor=None):
            images = images.to('cuda', self.dtype)
            result = {'latents': self.vae_encode(images)}
            if control_tensor is not None:
                result['control_latents'] = self.vae_encode(control_tensor.to('cuda', self.dtype))
            return result
        return fn

    def load_vision_image(self, path):
        pil_img = Image.open(path)
        if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
            pil_img = pil_img.convert('RGBA')
        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')
        scale = VISION_SHORT_EDGE / min(pil_img.size)
        if scale != 1.0:
            pil_img = ImageOps.scale(pil_img, scale)
        # comfy's qwen2vl preprocessing expects a [1, H, W, C] float tensor in 0..1
        # (it does its own resize/normalize); a PIL image just crashes it
        arr = torch.from_numpy(np.asarray(pil_img, dtype=np.float32)).div_(255.0)
        return arr.unsqueeze(0)

    def get_call_text_encoder_fn(self, text_encoder):
        def tokenize(text, images):
            kwargs = {'picture_label': PICTURE_LABEL}
            if images is not None:
                kwargs['images'] = images
            return text_encoder.tokenize(text, **kwargs)

        @torch.inference_mode()
        def fn(captions: list, is_video: list, control_files=None):
            assert not any(is_video)
            # resolve lazily: the text encoder is lazy-loaded and isn't
            # available when this fn is created (only once caching runs)
            images_list = []
            for i in range(len(captions)):
                imgs = None
                if control_files is not None and control_files[i] is not None:
                    cf = control_files[i]
                    if isinstance(cf, str):
                        cf = [cf]  # a single control file per example comes through as a string
                    imgs = [self.load_vision_image(f) for f in cf]
                images_list.append(imgs)

            # NOTE: no padding to a common length here. Each example is encoded
            # on its own, and get_conds pads to the batch max itself. Setting
            # tokenizer.min_length would actively break this: comfy pads every
            # text segment independently (picture label AND caption each padded
            # to min_length), so re-deriving min_length from the padded entries
            # makes the sequence grow exponentially example over example until OOM.

            # Encode one example at a time. The batched path doesn't work here:
            # comfy's embeds_info/vision-token tags only describe the LAST item
            # of a batched encode, and image token expansion can give edit
            # examples different expanded sequence lengths. Each fn output is a
            # list of per-example tensors (embeds (S,C), mask (S,), tags (S,)),
            # which the cache stores as one row per example.
            embeds_list = []
            mask_list = []
            tags_list = []
            have_tags = True
            for text, imgs in zip(captions, images_list):
                tokens = tokenize(text, imgs)
                tokens_dict = {}
                for k, v in tokens.items():
                    if isinstance(v, bool):
                        continue
                    tokens_dict[k] = [v[0]]
                o = text_encoder.encode_from_tokens_scheduled(tokens_dict)
                extra = o[0][1]
                embeds = o[0][0].to(self.dtype)
                if 'attention_mask' in extra:
                    attention_mask = extra['attention_mask']
                else:
                    attention_mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device=embeds.device)
                # to CPU right away: this fn runs per example in the caching
                # subprocess and everything here just gets pickled through the
                # task queue -- keeping GPU references alive per iteration makes
                # the caching allocator's reserved memory creep up example by example
                embeds_list.append(embeds.squeeze(0).cpu())
                mask_list.append(attention_mask.squeeze(0).cpu())
                # vision token tags, used to split the text span into modality runs for AdaLN
                tags = extra.get('minimax_token_tags')
                if tags is None:
                    have_tags = False
                elif have_tags:
                    tags_list.append(tags.cpu())

            result = {'text_embeds_0': embeds_list, 'attention_mask_0': mask_list}
            if have_tags:
                result['text_token_tags_0'] = tags_list
            # sequence lengths vary per example (vision token expansion), so the
            # allocator can't reuse its blocks cleanly -- hand the fragments back
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result

        return fn

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
        text_tags = inputs.get('text_token_tags_0', None)
        if text_tags is not None and text_tags.numel() > 0:
            text_tags = torch.stack(
                [torch.cat([u, u.new_full((max_seq_len - u.size(0),), -1, dtype=u.dtype)]) for u in text_tags]
            )
        elif text_tags is not None:
            text_tags = text_tags.new_empty((text_embeds.shape[0], 0), dtype=torch.long)
        return text_embeds, attention_mask, text_tags

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, f, h, w = latents.shape
        assert f == 1, 'MiniMax-H3-Image generates single images; use the minimax_h3 video model for videos'
        device = latents.device

        control_latents = inputs.get('control_latents', None)
        if control_latents is not None and control_latents.numel() > 0:
            assert control_latents.shape == latents.shape, (
                f'Control latents shape {control_latents.shape} does not match latents shape {latents.shape}'
            )
        else:
            control_latents = None

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)

        sigma_grid = self._get_sigma_grid()
        if sigma_grid is not None:
            # multi-head (PDD) checkpoint: each head was distilled on its own sigma, so training
            # draws uniformly from the grid. timestep_sample_method / shift / timestep_quantile
            # do not apply here.
            idx = torch.randint(0, len(sigma_grid) - 1, (bs,), device=device)
            t = torch.tensor(sigma_grid[:-1], device=device, dtype=torch.float32)[idx]
        else:
            # single-head checkpoint: generic flow-matching timestep sampling
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

            shift = self.model_config.get('shift', None)
            if shift:
                t = (t * shift) / (1 + (shift - 1) * t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, control_latents, t, *self.get_conds(inputs)), (target, mask)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model, self._get_sigma_grid())]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def get_loss_fn(self):
        @torch.autocast('cuda', enabled=False)
        def single_loss(output, target, mask=None):
            output = output.to(torch.float32)
            target = target.to(output.device, torch.float32)
            if 'huber_delta' in self.config:
                loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
            elif 'smooth_l1_beta' in self.config:
                loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
            else:
                loss = F.mse_loss(output, target, reduction='none')
            # empty tensor means no masking
            if mask is not None and mask.numel() > 0:
                mask = mask.to(output.device, torch.float32)
                loss *= mask
            return loss

        def loss_fn(output, label):
            if isinstance(output, (tuple, list)):
                output = output[0]
            target, mask = label
            return single_loss(output, target, mask).mean()
        return loss_fn

    @torch.no_grad()
    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def sample(self, w=512, h=512):
        lat_h, lat_w = h // self.spatial_compression, w // self.spatial_compression
        x = torch.randn((1, self.channels, 1, lat_h, lat_w), device='cuda')
        conds = self.conds
        sigma_grid = self._get_sigma_grid()
        if sigma_grid is not None:
            sigmas = sigma_grid
        else:
            sigmas = torch.cat([self.scheduler.timesteps / 1000.0, torch.zeros(1, device='cuda')]).tolist()
        for i in range(len(sigmas) - 1):
            t = torch.tensor([sigmas[i]], device='cuda', dtype=torch.float32)
            v = self.pipeline_model((x, None, t, *conds)).float()
            x = x + (sigmas[i + 1] - sigmas[i]) * v
        vae = self.get_vae()
        if isinstance(vae, nn.Module):
            vae = vae.to('cuda')
        else:
            vae.load_model_if_needed()
        img = self.vae_decode(x)
        if isinstance(vae, nn.Module):
            vae = vae.to('cpu')
        else:
            comfy.model_management.unload_all_models()
        if img.ndim == 5:
            img = img[:, :, 0]
        return img


class InitialLayer(nn.Module):
    def __init__(self, model, sigma_grid):
        super().__init__()
        if model.use_adaln_curves:
            self.adaln_t_table = model.adaln_t_table
        else:
            self.time_embedder = model.time_embedder
        self.video_patch_proj = model.video_patch_proj
        self.condition_proj = model.condition_proj
        self.rope = model.rope
        self.token_refiner = model.token_refiner
        self.model = [model]
        self.sigma_grid = sigma_grid
        self._cache = {}

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def get_layout_and_rope(self, text_len, lat_t, lat_h, lat_w, has_ref, dtype):
        refs = None
        if has_ref:
            refs = [{'kind': 'image', 'latent_h': lat_h, 'latent_w': lat_w}]
        sig = (text_len, lat_t, lat_h, lat_w, has_ref, dtype)
        layout = self._cache.get(sig)
        if layout is None:
            layout = PackedLayout(text_len, lat_t, lat_h, lat_w, 0, refs=refs)
            self._cache[sig] = layout
            device = torch.device('cuda')
            rope = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
            self._cache[('rope',) + sig] = rope
        return layout, self._cache[('rope',) + sig]

    def make_packed_sequence(self, video_x, control_x, t, context, context_mask, text_tags):
        assert video_x.shape[0] == 1  # needs batch dimension with batch size 1

        device = video_x.device
        dtype = context.dtype  # compute dtype

        lat_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        text_len = context.shape[1]
        has_ref = control_x is not None
        layout, rope_freqs = self.get_layout_and_rope(text_len, lat_t, lat_h, lat_w, has_ref, dtype)

        sigma = float(t.flatten()[0])
        t_v = 1.0 - sigma
        vis_aug = VISUAL_COND_TIMESTEP
        seg_t = {'text': t_v, 'video': t_v, 'ref_img': max(t_v, vis_aug)}
        unique_t = sorted({seg_t[kind] for _, _, kind in layout.segments})
        t_row = {v: i for i, v in enumerate(unique_t)}

        # per-token AdaLN mod rows (index into the batched AdalnProj output rows)
        mod_row = torch.empty(layout.seq_len, dtype=torch.long, device=device)
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == 'text' and text_tags is not None:
                # the text span mixes tags (vision pads carry the image modality), split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_row[a + run_start:a + i] = row_base + int(tags[run_start])
                        run_start = i
            else:
                mod_row[a:b] = row_base + SEG_TAG[kind]

        # embed
        img_update = layout.img_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        if has_ref:
            cond_rows = patchify_video(control_x.to(torch.float32), self.patch_size)
            # einsum in patchify_video downcasts under autocast, so match dtypes explicitly
            all_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=video_rows.dtype, device=device)
            all_rows[~img_update] = cond_rows.to(video_rows.dtype)
            all_rows[img_update] = video_rows
        else:
            all_rows = video_rows
        video_embed = self.video_patch_proj(all_rows).to(dtype)

        text_states = context
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states), transformer_options={})
        text_states = text_states[0]

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones((layout.seq_len,), dtype=context_mask.dtype, device=device)
        voff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == 'text':
                h[a:b] = text_states
                attention_mask[a:b] = context_mask
            else:  # ref_img / video (the target image is always the last segment)
                h[a:b] = video_embed[voff:voff + n]
                voff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            # adaln projections consume interpolated coordinates of the time-embedding curve
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        head_idx = 0
        if self.sigma_grid is not None:
            # the sigma this example was noised at maps to exactly one PDD head
            head_idx = min(range(len(self.sigma_grid) - 1), key=lambda k: abs(self.sigma_grid[k] - sigma))

        # pack as ints so we can pass as tensor between pipeline parallel layers
        va, vb, _ = layout.segments[-1]
        extra_ints = torch.tensor([va, vb, lat_t, lat_h, lat_w, head_idx], dtype=torch.int32, device=device)

        return h, attention_mask, t_emb, mod_row, rope_freqs, extra_ints

    def make_layer_inputs(self, video_x, control_x, t, context, context_mask, text_tags):
        h_list, attention_mask_list, t_emb_list, mod_row_list, extra_ints_list = [], [], [], [], []
        rope_freqs = None
        for i in range(video_x.shape[0]):
            cx = None if control_x is None else control_x[i:i + 1]
            tags = None if text_tags is None or text_tags.numel() == 0 else text_tags[i]
            h, attention_mask, t_emb, mod_row, rope, extra_ints = self.make_packed_sequence(
                video_x[i:i + 1], cx, t[i:i + 1], context[i:i + 1], context_mask[i], tags)
            if rope_freqs is None:
                rope_freqs = rope
            h_list.append(h)
            attention_mask_list.append(attention_mask)
            t_emb_list.append(t_emb)
            mod_row_list.append(mod_row)
            extra_ints_list.append(extra_ints)

        h = torch.stack(h_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)[:, None, None, :]  # 4D bool for SDPA
        t_emb = torch.stack(t_emb_list)
        mod_row = torch.stack(mod_row_list)
        extra_ints = torch.stack(extra_ints_list)
        outputs = make_contiguous(h, attention_mask, t_emb, mod_row, rope_freqs, extra_ints)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        video_x, control_x, t, *variable = inputs
        if control_x is not None and control_x.numel() == 0:
            control_x = None  # converted from None by split_batch
        if len(variable) == 2:
            context, context_mask = variable
            text_tags = None
        else:
            context, context_mask, text_tags = variable
            if text_tags is not None and text_tags.numel() == 0:
                text_tags = None
        return self.make_layer_inputs(video_x, control_x, t, context, context_mask, text_tags)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_row, rope_freqs, extra_ints = inputs

        self.offloader.wait_for_block(self.block_idx)

        h = self.layer(h, t_emb, mod_row, rope_freqs, attention_mask=attention_mask)

        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(h, attention_mask, t_emb, mod_row, rope_freqs, extra_ints)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_row, rope_freqs, extra_ints = inputs
        fl = self.final_layer
        va, vb, lat_t, lat_h, lat_w = extra_ints[0, :5].tolist()
        assert (extra_ints[:, :5] == extra_ints[0, :5].unsqueeze(0)).all()
        head_idx = extra_ints[:, 5].long()

        shift, scale = fl.adaln_proj(t_emb)
        target_mod_row = mod_row[:, va:vb]
        dim = h.shape[-1]
        gather_idx = target_mod_row.unsqueeze(-1).expand(-1, -1, dim)
        hv = (fl.norm(h[:, va:vb]) * (1.0 + scale.gather(1, gather_idx)) + shift.gather(1, gather_idx)).to(torch.float32)

        n = fl.pdd_steps
        video_dim = fl.video_dim
        bs = h.shape[0]
        out = torch.empty(bs, vb - va, video_dim, dtype=torch.float32, device=h.device)
        W = fl.video_out.weight.to(torch.float32)  # (video_dim * n, hidden) head bank; may be float8 on float8 checkpoints
        bias = None if fl.video_out.bias is None else fl.video_out.bias.to(torch.float32)
        for k in head_idx.unique().tolist():
            sel = head_idx == k
            if n == 1:
                v = F.linear(hv[sel], W, bias)
            else:
                # row block 0 is a full head, later blocks are offsets from it
                Wv = W.view(n, video_dim, -1)
                Bv = bias.view(n, video_dim) if bias is not None else None
                k = int(k)
                Wk = Wv[0] if k == 0 else Wv[0] + Wv[k]
                Bk = None if Bv is None else (Bv[0] if k == 0 else Bv[0] + Bv[k])
                v = F.linear(hv[sel], Wk, Bk)
            out[sel] = v.to(torch.float32)

        vid = unpatchify_video(out, lat_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        return -vid
