"""Full ONNX-only StyleTTS2 inference (no PyTorch at runtime).

Requires three ONNX files produced by export_onnx.py and export_onnx_style.py:
  styletts.onnx           — main TTS synthesiser
  styletts_bert.onnx      — PLBERT text encoder
  styletts_denoiser.onnx  — diffusion denoiser (single step)

And the config produced by export_onnx_style.py:
  styletts_config.json    — vocab + style_dim

Pipeline:
  1. Tokenise phonemised text
  2. BERT ONNX  →  bert_embedding
  3. Karras schedule + ADPM2 sampling loop (pure numpy, calls denoiser ONNX)
  4. TTS ONNX  →  waveform
  5. Write WAV

Usage:
  python run_onnx.py -t "hɛloʊ, haʊ ɑːɹ juː?" -o output.wav
"""

import argparse
import json
import math
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile as wavfile
from pathlib import Path

from nltk.tokenize import word_tokenize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')


# ── Tokeniser (pure Python, no PyTorch needed) ────────────────────────────────

def build_vocab(vocab_list):
    return {ch: i for i, ch in enumerate(vocab_list)}

def encode(text: str, vocab_dict: dict) -> np.ndarray:
    return np.array([vocab_dict[c] for c in text if c in vocab_dict], dtype=np.int64)


# ── Karras schedule (mirrors KarrasSchedule in sampler.py) ───────────────────

def karras_sigmas(num_steps: int, sigma_min=0.0001, sigma_max=3.0, rho=9.0) -> np.ndarray:
    rho_inv = 1.0 / rho
    steps   = np.arange(num_steps, dtype=np.float64)
    sigmas  = (
        sigma_max ** rho_inv
        + (steps / (num_steps - 1)) * (sigma_min ** rho_inv - sigma_max ** rho_inv)
    ) ** rho
    sigmas  = np.append(sigmas, 0.0).astype(np.float32)
    return sigmas


# ── ADPM2 sampler (mirrors ADPM2Sampler in sampler.py) ───────────────────────

def _adpm2_get_sigmas(sigma: float, sigma_next: float, rho: float = 1.0):
    sigma_up   = math.sqrt(max(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2, 0.0))
    sigma_down = math.sqrt(max(sigma_next**2 - sigma_up**2, 0.0))
    sigma_mid  = ((sigma ** (1.0 / rho) + sigma_down ** (1.0 / rho)) / 2.0) ** rho
    return sigma_up, sigma_down, sigma_mid


def sample_style(
    denoiser_session: ort.InferenceSession,
    embedding_np: np.ndarray,
    noise_np: np.ndarray,
    num_steps: int = 10,
) -> np.ndarray:
    """Run ADPM2 diffusion sampling and return the style vector.

    Args:
        denoiser_session: ONNX session for styletts_denoiser.onnx
        embedding_np:     BERT embedding, shape [1, seq_len, hidden_size]
        noise_np:         Starting noise, shape [1, 1, style_dim*2]
        num_steps:        Number of diffusion steps (default 10)

    Returns:
        style vector, shape [1, 1, style_dim*2]
    """
    sigmas = karras_sigmas(num_steps)
    x = (sigmas[0] * noise_np).astype(np.float32)

    def denoise(x_in: np.ndarray, sigma_val: float) -> np.ndarray:
        sigma_arr = np.array([sigma_val], dtype=np.float32)
        (x_out,) = denoiser_session.run(
            ['x_denoised'],
            {'x_noisy': x_in, 'sigma': sigma_arr, 'embedding': embedding_np},
        )
        return x_out

    for i in range(num_steps - 1):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        sigma_up, sigma_down, sigma_mid = _adpm2_get_sigmas(sigma, sigma_next)

        x_denoised = denoise(x, sigma)
        d          = (x - x_denoised) / sigma

        x_mid          = x + d * (sigma_mid - sigma)
        x_denoised_mid = denoise(x_mid, sigma_mid)
        d_mid          = (x_mid - x_denoised_mid) / sigma_mid

        x = x + d_mid * (sigma_down - sigma)
        if sigma_up > 0.0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * sigma_up

    # DiffusionSampler has clamp=False, so no clamping
    return x  # shape [1, 1, style_dim*2]


# ── Inference pipeline ────────────────────────────────────────────────────────

def run(args):
    # Load vocab / metadata
    with open(args.config_json, encoding='utf-8') as f:
        meta = json.load(f)
    vocab_dict = build_vocab(meta['vocab'])
    style_dim  = meta['style_dim']   # 128  →  style_dim*2 = 256


    ps = global_phonemizer.phonemize([args.text])
    ps = ' '.join(word_tokenize(ps[0]))
    # Tokenise
    tokens = encode(ps, vocab_dict)          # [L]
    if tokens.size == 0:
        raise ValueError('No tokens produced — make sure the text is phonemised and uses the model vocabulary.')

    tokens_batched = tokens.reshape(1, -1)           # [1, L]
    attn_mask      = np.ones_like(tokens_batched, dtype=np.int32)

    # ONNX sessions
    sess_opts        = ort.SessionOptions()
    bert_sess        = ort.InferenceSession(args.bert_model,     sess_options=sess_opts)
    denoiser_sess    = ort.InferenceSession(args.denoiser_model, sess_options=sess_opts)
    tts_sess         = ort.InferenceSession(args.tts_model,      sess_options=sess_opts)

    # 1. BERT encoding
    (bert_emb,) = bert_sess.run(
        ['bert_embedding'],
        {'tokens': tokens_batched, 'attention_mask': attn_mask},
    )  # [1, L, hidden_size]

    # 2. Style prediction via diffusion sampling
    noise = np.random.randn(1, 1, style_dim * 2).astype(np.float32)
    style = sample_style(denoiser_sess, bert_emb, noise, num_steps=args.diffusion_steps)
    # style: [1, 1, 256]  →  squeeze to [1, 256]
    s_prev = style.squeeze(1)   # [1, 256]

    # 3. TTS synthesis
    # The exported styletts.onnx takes 1-D tokens (model.forward prepends 0-token internally)
    (wav_out,) = tts_sess.run(
        ['output_wav'],
        {
            'tokens': tokens,                                        # [L]
            'speed':  np.array(args.speed, dtype=np.float32),
            's_prev': s_prev,                                        # [1, 256]
        },
    )  # [T]

    # 4. Write WAV
    wav_out  = wav_out.astype(np.float32)
    peak     = np.abs(wav_out).max()
    if peak > 0.0:
        wav_out /= peak
    wav_i16  = (wav_out * 32767).astype(np.int16)
    wavfile.write(args.output, 24000, wav_i16)
    print(f'Saved {args.output}  ({wav_i16.shape[0] / 24000:.2f}s)')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX-only StyleTTS2 inference')
    parser.add_argument('-t',  '--text',             type=str,   required=True,
                        help='Input text')
    parser.add_argument('-o',  '--output',            type=str,   default='output.wav',
                        help='Output WAV path (default: output.wav)')
    parser.add_argument('--tts_model',               type=str,   default='styletts.onnx')
    parser.add_argument('--bert_model',              type=str,   default='styletts_bert.onnx')
    parser.add_argument('--denoiser_model',          type=str,   default='styletts_denoiser.onnx')
    parser.add_argument('--config_json',             type=str,   default='styletts_config.json')
    parser.add_argument('--speed',                   type=float, default=1.0)
    parser.add_argument('--diffusion_steps',         type=int,   default=10)
    parser.add_argument('--seed',                    type=int,   default=None,
                        help='Fix numpy RNG seed for reproducible style generation')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    run(args)
