# Resume Skill Scorer — SFT (QLoRA on Phi-3-medium)

This repo fine-tunes `microsoft/Phi-3-medium-128k-instruct` with **SFT** using **QLoRA** to score resumes on a few skills and return a brief rationale.

## TL;DR

- **Base model:** `microsoft/Phi-3-medium-128k-instruct`
- **Method:** SFT with QLoRA (4-bit NF4) via `trl`’s `SFTTrainer`
- **Data:** Synthetic instruction–response pairs generated with **Claude**
- **Context length used:** 1,024 tokens
- **Hardware:** 4-bit loading + gradient checkpointing (fits a single GPU)
- **Result (quick sanity set):** **66.7%** (4/6 categories within expected range)
- **Next step:** Try **DPO** to align outputs to preference targets

---

## Dataset

- **Source:** Programmatically generated with Claude (anthropic).  
- **Task format:** Given a resume snippet, the model outputs per-skill scores (0–5) + short reasoning.

> ⚠️ **Note on data provenance:** Synthetic labels were produced by a third-party LLM. If you plan to release the dataset, document licensing/terms and include a note about synthetic labels.

---

## Training Setup

**Quantization:** 4-bit NF4 via `bitsandbytes`  
**PEFT / LoRA:**
- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Bias: `none`
- Task: `CAUSAL_LM`

**Trainer / Args:**
- `epochs=3`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `gradient_checkpointing=True`
- `learning_rate=2e-4`, `weight_decay=0.01`
- `lr_scheduler_type=cosine`, `warmup_steps=10`
- `fp16=True`
- `logging_steps=10`, `eval_steps=50`, `save_steps=100`
- `report_to=none`

**Tokenization:**
- `tokenizer.pad_token = tokenizer.eos_token`
- `tokenizer.model_max_length = 1024`

**Formatting:**
- For `format_type="text"` the trainer uses a simple `formatting_func` that returns the `"text"` field (pre-combined prompt + target).

---

## Code Entry Point

```python
def setup_and_train_model(
    train_dataset,
    val_dataset,
    format_type="text",
    model_name="microsoft/Phi-3-medium-128k-instruct",
    output_dir=f"{BASE_PATH}/fine-tuned-model_Phi-3_medium"
):
    ...
```

- Loads the base model in **4-bit**, prepares for k-bit training, applies **LoRA**, and launches **SFTTrainer**.
- Saves merged adapter weights + tokenizer to `output_dir`.

---

## Quick Evaluation (Sanity Set)

The evaluation compares model-predicted skill scores (0–5) against a **target expected range** per category.  
A category counts as **correct** if the predicted score falls within the expected range.

| Skill            | Predicted | Expected |
|------------------|-----------|----------|
| Python           | 4         | 4–5      |
| Machine Learning | 4         | 4–5      |
| PyTorch          | 3         | 3–4      |
| Leadership       | 2         | 2–3      |
| Java             | 2         | 0–1 ❌   |
| Communication    | 1         | 2–3 ❌   |

**Accuracy:** 4/6 = **66.7%**  
**Verdict:** 🟡 **MODERATE ACCURACY** — model is broadly calibrated but shows:
- **False positive** on **Java** (over-predicting)
- **Under-scoring** **Communication**

---

## Why DPO Next?

SFT teaches the base format/style but doesn’t strongly enforce **preferences** (e.g., “don’t over-credit Java,” “weigh communication evidence higher”).  
**DPO** (or PPO/GRPO) uses **preference pairs** to directly optimize for the **better** response, which should:
- Reduce **over-scoring** on weakly-supported skills (Java)
- Improve **calibration** for soft skills (Communication)
- Tighten alignment to your rubric without RL infrastructure overhead (DPO is simpler than PPO)

**Minimal DPO plan:**
1. Collect preference pairs (A better than B) for the same resume input.
2. Train with `trl`’s `DPOTrainer` using your SFT model as the reference policy.
3. Re-check the same sanity set + a larger validation set.


---

## Known Limitations

- Synthetic labels may encode annotator-LLM bias.
- Scores on soft skills (e.g., Communication) are sensitive to prompt phrasing and evidence density.
- Using `model_max_length=1024` truncates long resumes; consider chunking for longer CVs.

---

## Roadmap

- ✅ SFT (this run)
- 🔜 **DPO** with preference pairs
- 🔜 Add **chunking** + aggregation for long resumes
- 🔜 Expand eval to a larger, hand-checked set with per-skill calibration plots
