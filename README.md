# StyleTTS2 Inference library

This is a **StyleTTS2 inference library** — an inference-only fork of StyleTTS2 focused on PyTorch and ONNX inference. Training code has been removed.

## How to use with pytorch

```
import soundfile
from nltk.tokenize import word_tokenize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')


from styletts2_inference.models import StyleTTS2

model = StyleTTS2(hf_path='patriotyk/StyleTTS2-LibriTTS', device='cpu')

text = 'Hello, how are you?'
ps = global_phonemizer.phonemize([text])
ps = ' '.join(word_tokenize(ps[0]))
tokens = model.tokenizer.encode(ps)

style = model.predict_style_multi('prompt.wav', tokens)

wav = model(tokens, s_prev=style)
soundfile.write('gennnerated.wav', wav.cpu().numpy(), 24000)

```

## How to use with onnx

First you need to export model to onnx format using included script `export_onnx.py` and `export_onnx_style.py`. This script will generate

`styletts.onnx`          — The full model
`styletts_bert.onnx`      — PLBERT: (tokens, attention_mask) -> bert_embedding
`styletts_denoiser.onnx`  — KDiffusion denoiser: (x_noisy, sigma, embedding) -> x_denoised
`styletts_config.json`    — vocab + minimal metadata for the inference script

files in the current directory.
Then you can infer it using `run_onnx.py`.

For multispeaker, you have to generate `style` vector from audio file and pass it to onnx session. 
The ONNX export has been tested on a single-speaker model.