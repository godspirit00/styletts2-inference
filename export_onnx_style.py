"""Export PLBERT and diffusion denoiser to ONNX.

Produces:
  styletts_bert.onnx      — PLBERT: (tokens, attention_mask) -> bert_embedding
  styletts_denoiser.onnx  — KDiffusion denoiser: (x_noisy, sigma, embedding) -> x_denoised
  styletts_config.json    — vocab + minimal metadata for the inference script

Usage (HuggingFace):
  python export_onnx_style.py -hf patriotyk/StyleTTS2-LibriTTS -t "hɛloʊ"

Usage (local):
  python export_onnx_style.py -c Checkpoints/config.yml -w Checkpoints/model.pth -t "hɛloʊ"
"""

import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from styletts2_inference.models import StyleTTS2


# ── Wrapper modules ────────────────────────────────────────────────────────────

class PLBertONNX(nn.Module):
    """Thin wrapper so PLBert exports with named inputs/outputs.

    attention_mask is cast to float32 inside the wrapper because
    HuggingFace's get_extended_attention_mask does (1.0 - mask) which
    produces incorrect ONNX graphs when the mask is an integer type.
    """
    def __init__(self, plbert):
        super().__init__()
        self.plbert = plbert

    def forward(self, tokens, attention_mask):
        return self.plbert(tokens, attention_mask.float())


class DenoiserONNX(nn.Module):
    """Wraps a single KDiffusion denoise step for ONNX export.

    Reproduces KDiffusion.get_scale_weights + net forward, bypassing the
    Transformer1d CFG / classifier-free-guidance path (embedding_scale=1.0
    always used in predict_style_single, so only the real embedding branch
    is needed).

    Inputs:
        x_noisy   : [1, 1, style_dim*2]
        sigma     : [1]  float
        embedding : [1, seq_len, hidden_size]
    Output:
        x_denoised: [1, 1, style_dim*2]
    """
    def __init__(self, transformer_net: nn.Module, sigma_data: float):
        super().__init__()
        self.net = transformer_net
        self.register_buffer('sigma_data', torch.tensor(sigma_data, dtype=torch.float32))

    def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor, embedding: torch.Tensor):
        # Reproduce KDiffusion.get_scale_weights
        c_noise = torch.log(sigma) * 0.25           # [1]
        s3d = sigma.view(1, 1, 1)                   # [1, 1, 1]
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (s3d ** 2 + sd2)
        c_out  = s3d * self.sigma_data * (sd2 + s3d ** 2) ** -0.5
        c_in   = (s3d ** 2 + sd2) ** -0.5

        # Call the transformer directly (no CFG double-pass needed at scale=1)
        x_pred = self.net.run(c_in * x_noisy, c_noise, embedding, None)
        return c_skip * x_noisy + c_out * x_pred


# ── Export helpers ─────────────────────────────────────────────────────────────

def export_bert(model: StyleTTS2, tokens_batched: torch.Tensor):
    seq_len = tokens_batched.shape[1]
    attention_mask = torch.ones(1, seq_len, dtype=torch.int32)

    wrapper = PLBertONNX(model.plbert).eval()

    torch.onnx.export(
        wrapper,
        args=(tokens_batched, attention_mask),
        f='styletts_bert.onnx',
        dynamo=False,
        export_params=True,
        input_names=['tokens', 'attention_mask'],
        output_names=['bert_embedding'],
        training=torch.onnx.TrainingMode.EVAL,
        opset_version=17,
        dynamic_axes={
            'tokens':         {1: 'seq_len'},
            'attention_mask': {1: 'seq_len'},
            'bert_embedding': {1: 'seq_len'},
        },
    )
    print('Exported styletts_bert.onnx')


def export_denoiser(model: StyleTTS2, tokens_batched: torch.Tensor):
    kdiffusion   = model.sampler.diffusion           # KDiffusion
    transformer  = kdiffusion.net                    # Transformer1d
    sigma_data   = float(kdiffusion.sigma_data)

    style_dim2  = model.config.model_params.style_dim * 2   # 256
    hidden_size = model.config.plbert_params.hidden_size    # e.g. 768
    seq_len     = tokens_batched.shape[1]

    wrapper  = DenoiserONNX(transformer, sigma_data).eval()
    x_dummy  = torch.randn(1, 1, style_dim2)
    sig_dummy = torch.tensor([1.0])
    emb_dummy = torch.randn(1, seq_len, hidden_size)

    torch.onnx.export(
        wrapper,
        args=(x_dummy, sig_dummy, emb_dummy),
        f='styletts_denoiser.onnx',
        dynamo=False,
        export_params=True,
        input_names=['x_noisy', 'sigma', 'embedding'],
        output_names=['x_denoised'],
        training=torch.onnx.TrainingMode.EVAL,
        opset_version=17,
        dynamic_axes={
            'embedding': {1: 'seq_len'},
        },
    )
    print('Exported styletts_denoiser.onnx')


def save_config(model: StyleTTS2):
    """Save vocab + shape metadata so the inference script needs no PyTorch."""
    metadata = {
        'vocab':     list(model.config.model_params.vocab),
        'style_dim': int(model.config.model_params.style_dim),
    }
    with open('styletts_config.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    print('Saved styletts_config.json')


# ── Main ───────────────────────────────────────────────────────────────────────

def export(args):
    if args.hf_path:
        model = StyleTTS2(hf_path=args.hf_path, device='cpu')
    else:
        model = StyleTTS2(config_path=str(args.config),
                          weights_path=str(args.weights_path), device='cpu')
    model.eval()

    tokens_batched = model.tokenizer.encode(args.text).unsqueeze(0)  # [1, L]

    export_bert(model, tokens_batched)
    export_denoiser(model, tokens_batched)
    save_config(model)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export StyleTTS2 PLBERT and diffusion denoiser to ONNX'
    )
    parser.add_argument('-hf', '--hf_path',      type=str,  default=None,
                        help='HuggingFace repo id, e.g. patriotyk/StyleTTS2-LibriTTS')
    parser.add_argument('-c',  '--config',        type=Path, default=None,
                        help='Path to config YAML (must include plbert_params and vocab)')
    parser.add_argument('-w',  '--weights_path',  type=Path, default=None,
                        help='Path to .pth checkpoint')
    parser.add_argument('-t',  '--text',          type=str,  required=True,
                        help='Phonemized dummy text used for tracing (e.g. "hɛloʊ")')
    args = parser.parse_args()

    if not args.hf_path and not (args.config and args.weights_path):
        parser.error('Provide --hf_path OR both --config and --weights_path')

    export(args)
