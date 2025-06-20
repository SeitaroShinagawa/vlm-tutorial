# vlm-tutorial

本リポジトリでは、Vision-Language Model (VLM)のチュートリアルとして、コードを読みながら実装について解説します。

## Google Colabの利用方法

<a href="" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## エンコーダ型VLM

### CLIPの推論

CLIPの基本的な推論方法について説明する他、Typographic Attack, Visual promptingの効果についても確認します。

[OpenCLIP](https://github.com/mlfoundations/open_clip)を使います。

Colab: https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb

[SigLIP](https://www.isus.jp/wp-content/uploads/openvino/2024/docs/notebooks/siglip-zero-shot-image-classification-with-output.html)

### CLIPの学習

CLIPの学習を行います。性能が高いSigLIPを用います。（SigLIPについては別途説明します）


## デコーダ型VLM

### デコーダVLMの推論

#### Visual Proompting

SAMを使った方法


### 教師あり学習（Supervised Fine-Tuning: SFT）: Visual Instruction Tuning

[Gemma3の量子化モデル（Unsloth）](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=hzHNg-MmWfZ6)
https://github.com/unslothai/unsloth
TRLによるSFT

ValueError: Your setup doesn't support bf16/gpu. You need Ampere+ GPU with cuda>=11.0
SFTTrainerの引数で`fp16=True`
Unslothだとfp16はサポートされてないらしい、詰んだかも
→だめ

PaliGemma
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

smalvlm
ImportError: /usr/local/lib/python3.11/dist-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

### 強化学習：DPO (Direct Preference Optimization)


