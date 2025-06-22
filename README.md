# vlm-tutorial

本リポジトリでは、Vision-Language Model (VLM)のチュートリアルとして、コードを読みながら実装について解説します。

<a href="" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

SAMを使った方法


### 教師あり学習（Supervised Fine-Tuning: SFT）: Visual Instruction Tuning

HuggingfaceのOpen-Source AI Cookbookを利用
https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl

- T4で実行する際の注意：flash-attentionは使わない（Ampare世代のGPUの場合は利用可能、高速化やメモリの削減が見込める）
- `_attn_implementation=None`とする（A100などを使えるときは``）



### 強化学習：DPO (Direct Preference Optimization)

HuggingfaceのOpen-Source AI Cookbookを利用
https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct



loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
 [ 222/2830 2:03:29 < 24:23:53, 0.03 it/s, Epoch 0.08/1]
Step	Training Loss
25	1.718100
50	0.794400
75	0.209100
100	0.158000
125	0.119200
150	0.124000
175	0.109300
200	0.101000


## VLMによる評価

llm-jp-eval-mmを使って評価
OpenAI APIは使えない（おそらく）

localのLLMで評価できるようにしたい→vllmでサーバを立てて実行
拡張の必要あり

自前で作成したデータセットの利用


visual prompting 
matplotlibで区分け、番号付けする
SAM2を使ってセマンティックセグメンテーションしてからラベル付け

推論にはセグメンテーション付けした後のやつを使う


llm as a judge

# テスト

lmm-jp-eval-mmの拡張
以下の方法で動作チェックをする予定ですが、vllmのインストールにはGPUが必要なので実行不要です。テストだけ書いておいてください。

```
from PIL import Image
from eval_mm import TaskRegistry, ScorerRegistry, ScorerConfig
from vllm_as_a_judge import 

score_registry = ScorerRegistry()
score_registry_scorers["vLLM-as-a-judge"]=

class MockVLM:
    def generate(self, images: list[Image.Image], text: str) -> str:
        return "宮崎駿"

task = TaskRegistry.load_task("japanese-heron-bench")
example = task.dataset[0]

input_text = task.doc_to_text(example)
images = task.doc_to_visual(example)
reference = task.doc_to_answer(example)

model = MockVLM()
prediction = model.generate(images, input_text)

scorer = ScorerRegistry.load_scorer(
    "vLLM-as-a-judge",
    ScorerConfig(docs=task.dataset)
)
result = scorer.aggregate(scorer.score([reference], [prediction]))
print(result)
# AggregateOutput(overall_score=5.128205128205128, details={'rougel': 5.128205128205128})
```

google colabでuvを使う場合は`uv run`経由でしかpythonが実行できないので要注意

lmm-jp-eval-mmのpyproject.tomlのflash-attnは削除する（T4では対象外でエラーとなる）

推論の実行
uv run --group vllm_normal python examples/sample_vllm.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --task_id japanese-heron-bench  \
  --result_dir result  \
  --metrics vllm_as_a_judge \
  --judge_model Qwen/Qwen2.5-VL-3B-Instruct \
  --inference_only \
  --overwrite

base_vllm.pyのengine_args_dictは`tensor_parallel_size: 1`にする
vllm_registry.pyの`get_engine_config‘を変更
```python
    def get_engine_config(self, model_id: str) -> dict:
        return {
            "max_model_len": 2048,
            "max_num_seqs": 1,
            "limit_mm_per_prompt": {"image": 1},
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.8,
        }
  ```