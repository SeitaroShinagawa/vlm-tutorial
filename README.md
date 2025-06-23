# vlm-tutorial

本リポジトリでは、Vision-Language Model (VLM)のチュートリアルとして、コードを読みながら実装について解説します。

## エンコーダ型VLM

### CLIPの推論

CLIPの基本的な推論方法について説明する他、Visual promptingの効果についても確認します。

<a href="https://colab.research.google.com/drive/1kcu4KywaKuEFb1sYAnKjHz017YQ5Wg9T?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

参考URL：
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SigLIP](https://www.isus.jp/wp-content/uploads/openvino/2024/docs/notebooks/siglip-zero-shot-image-classification-with-output.html)


### CLIPの学習

CLIPの学習方法について実装をみながら確認します。  
注意：既存のCoalb Notebookに一部手を入れたものになります。

<a href="https://colab.research.google.com/drive/1nrTbeB3_mokXNWWgvQaCqlx1bobXZtVl?usp=sharing
" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## デコーダ型VLM

### デコーダVLMの推論

#### Visual Proompting

SAM2を用いた領域分割と領域ごとの番号付けの効果を確認します。

SAM2による領域分割

<a href="https://colab.research.google.com/drive/1cMVE7KSwaPtqioFrIja9OtxQHxTeEbxM?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

vLLMによる、Qwen/Qwen2.5-VL-3B-Instructの実行（vllm serveによりサーバを立てて接続）  

<a href="https://colab.research.google.com/drive/1IY0Y_fT4bUisw90Ju_LKzwB38t9sp-B0?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### デコーダVLMの学習

#### 教師あり学習（Supervised Fine-Tuning: SFT）: Visual Instruction Tuning

HuggingfaceのOpen-Source AI Cookbookを利用
本家：https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl

<a href="https://colab.research.google.com/drive/1_oh8MWt1zW0F7Cc3q9ZUiEHOn6QUCuRa?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


#### 強化学習：DPO (Direct Preference Optimization)

HuggingfaceのOpen-Source AI Cookbookを利用
本家：https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct

<a href="https://colab.research.google.com/drive/1LImRky1TG2VcZ88scYVpYeLXAudJ6z7m?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


