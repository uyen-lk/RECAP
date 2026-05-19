
# RECAP: A Robustness Evaluation Framework for Forgotten Knowledge Recovery in LLMs Unlearning

**RECAP** (**Re**covery via **C**omprehensive **A**dversarial **P**robing), a robustness evaluation framework that systematically assesses whether unlearned models remain vulnerable to adversarial recovery of forgotten knowledge through both recall and recognition pathways using black-box probes.
RECAP supports two unlearning benchmarks:

- **TOFU** — fictional author biography knowledge (open-ended QA)
- **WMDP** — hazardous knowledge (multiple-choice)


## 📍 Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Evaluation Scripts](#2-evaluation-scripts)
   - [2.1 Recall-based Recovery (TOFU): Open-ended QA under Adversarial Attacks](#31-tofu-open-ended-qa-under-adversarial-attacks)
   - [2.2 Recognition-based Recovery (WMDP-Cyber): Multiple-Choice Accuracy under Adversarial Attacks](#32-wmdp-cyber-multiple-choice-accuracy-under-adversarial-attacks)
3. [Attack Types](#3-attack-types)
4. [Prompting Strategies](#4-prompting-strategies)
5. [Running Experiments](#5-experiments)
6. [Citing Our Work](#6-citing-our-work)

---



## ⚡ 1. Environment Setup
```
#Environment setup
conda create -n probe-unlearn python=3.11
conda activate probe-unlearn
pip install -r requirements.txt

mkdir log
```

If your unlearned model is hosted on a private HuggingFace Hub repository, export your token before running:

```
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

## 🚀 2. Evaluation Scripts

### 2.1 Recall-based Recovery (TOFU): Open-ended QA under Adversarial Attacks

Loads TOFU questions, generates free-form responses under each attacker × strategy combination, then computes text-similarity and semantic-similarity metrics against the ground-truth answer.

Higher similarity = more unlearned knowledge recovered = more effective attack.

| Metric | Type |
|---|---|
| ROUGE-L Recall | Text similarity |
| BLEU | Text similarity |
| METEOR | Text similarity |
| SBERT| Semantic similarity |
| USECS | Semantic similarity |
| BERTScore F1 | Semantic similarity |

```
python src/tofu_attack.py \
    --model <huggingface_model_id_or_local_path> \
    --data_path data/dataset_tofu.csv \
    --output_dir output \
    --apply_chat_template \
    --strategies zero_shot few_shot multiturn \
    --dtype auto \
    --batch_size 32
```


**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | — | HuggingFace model ID or local checkpoint path |
| `--data_path` | `data/dataset.csv` | TOFU CSV with `question` and `answer` columns |
| `--output_dir` | `output_tofu_attack` | Root directory for results |
| `--attackers` | all | Subset of attackers to run: `DAN`, `prefix_injection`, `pretending`, `refusal_suppression`, `step_jailbreaking`, `fill_in_the_blank` |
| `--strategies` | `zero_shot few_shot multiturn` | Prompting strategies (see [§4](#4-prompting-strategies)) |
| `--apply_chat_template` | on | Apply the tokenizer's chat template to prompts |
| `--dtype` | `auto` | Model dtype: `auto`, `bf16`, `fp16`, `fp32` |
| `--batch_size` | `32` | Generation batch size |
| `--max_new_tokens` | `218` | Maximum new tokens to generate per response |
| `--device` | auto | `cuda` or `cpu` |
| `--overwrite` | off | Re-run and overwrite existing results |
| `--sbert_model` | `all-mpnet-base-v2` | SentenceBERT checkpoint |
| `--use_model` | `distiluse-base-multilingual-cased-v1` | USECS checkpoint |
| `--bertscore_model` | `microsoft/deberta-xlarge-mnli` | BERTScore backbone |

### 2.2 Recognition-based Recovery (WMDP-Cyber): Multiple-Choice Accuracy under Adversarial Attacks

Wraps each WMDP-Cyber question in an adversarial attack prefix, then measures multiple-choice accuracy by scoring next-token log-probabilities for answer letters A/B/C/D. Directly measures whether an attack causes the model to answer hazardous cybersecurity questions correctly.

```
python src/wmdp_attack.py \
    --model <huggingface_model_id_or_local_path> \
    --output_dir output \
    --apply_chat_template \
    --strategies zero_shot few_shot multiturn \
    --dtype auto \
    --batch_size 32
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | — | HuggingFace model ID or local checkpoint path |
| `--output_dir` | `output` | Root directory for results |
| `--attackers` | all | Subset of attackers: `DAN`, `prefix_injection`, `pretending`, `refusal_suppression`, `step_jailbreaking`, `indirect_negation` |
| `--strategies` | `zero_shot few_shot multiturn` | Prompting strategies (see [§4](#4-prompting-strategies)) |
| `--apply_chat_template` | on | Apply the tokenizer's chat template |
| `--dtype` | `auto` | Model dtype: `auto`, `bf16`, `fp16`, `fp32` |
| `--batch_size` | `32` | Batch size for forward passes |
| `--device` | auto | `cuda` or `cpu` |
| `--overwrite` | off | Re-run and overwrite existing results |


## ❌ 3. Attack Types

Both evaluation scripts share the same family of black-box jailbreaking attacks. Templates are embedded directly in the evaluation scripts.

| Attacker | Description | TOFU | WMDP |
|---|---|:---:|:---:|
| `DAN` | "Do Anything Now" — instructs the model to act as an unrestricted AI that ignores all safeguards and forgotten knowledge | ✓ | ✓ |
| `prefix_injection` | Forces responses to begin with `"Sure! Here's what I found"` | ✓ | ✓ |
| `pretending` | Role-plays as a domain expert (literary biographer for TOFU; cybersecurity researcher for WMDP) | ✓ | ✓ |
| `refusal_suppression` | Instructs the model to never produce refusals or apologies | ✓ | ✓ |
| `step_jailbreaking` | Step-wise instructions nudging the model to read the question, ignore safety rules, and answer directly | ✓ | ✓ |
| `fill_in_the_blank` | Uses the model itself to rephrase each question into an incomplete declarative sentence, then completes it (bypasses Q&A refusal by framing as sentence completion) | ✓ | — |
| `indirect_negation` | Frames the task as identifying which three options are wrong, not which one is right | — | ✓ |

---



## 💬 4. Prompting Strategies

Each attacker is applied under multiple prompting strategies to study how presentation affects attack success.

| Strategy | Description |
|---|---|
| `zero_shot` | Attack prefix prepended directly to the question in a single user turn |
| `few_shot` | 2 benign domain-relevant Q&A examples prepended before the attack prefix + question |
| `multiturn` | True two-turn dialogue: the model generates a real Turn 1 reply to an attacker-specific setup message. Turn 2 then asks the target question. One Turn 1 generation is reused across all questions for the same attacker.| 
| `context_injection` | For base model such as Phi-1.5 with no chat template, a plain-text "written history" format is embedded before the question to mimic multi-turn prompting. |

---

## 💻 5. Running Experiments
```
sbatch scripts/batch_tofu.sh
sbatch scripts/batch_wmdp.sh
```

## 📝 6. Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
```










