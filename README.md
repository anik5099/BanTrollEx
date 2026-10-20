# BanTrollEx: Who, How, and Why? Explaining Bengali Troll Memes with Vision-Language Models

> **Content warning:** This repository contains memes that may be offensive or sensitive to some readers. They are included only for research on understanding harmful online content.

A troll meme makes fun of someone by combining an image with a short caption. Often neither part is offensive on its own; the ridicule comes from how they work together with local culture. Existing troll meme datasets give only a single **troll / not-troll** label, which hides *whom* the meme attacks, *how* it attacks, and *why* it counts as trolling.

**BanTrollEx** is the first Bengali troll meme dataset, and to our knowledge the first troll meme resource in any language that annotates all three together:

| Task | Question | Labels |
|------|----------|--------|
| **Target identification** | *Who* is attacked? | Individual, Organization, Community, Concept |
| **Mechanism identification** | *How* does the meme troll? | Direct Derision, Comparative/Stereotypical Ridicule, Ironic/Deceptive Framing, Provocative Baiting |
| **Explanation generation** | *Why* is it trolling? | Human-written free-text explanation (English) |

We use BanTrollEx to benchmark **ten open-source vision-language models (2B–12B)** zero-shot, using a **gold-label conditioning** protocol that measures how much knowing the target and the mechanism helps a model explain a meme.

---

## Key Findings

- **Explaining troll memes is hard.** The best configuration (Qwen3.5-9B with both gold labels) reaches a mean human rating of only **3.67 / 5**. No model reaches *good* (4) in any setting.
- **Gold labels help, and the mechanism matters more than the target.** Giving both labels raises the mean human rating from **2.97 to 3.26**. The gold mechanism alone adds +0.09; the gold target alone adds only +0.02.
- **Small models gain the most.** Gold labels narrow the gap between medium (8–12B) and small (≤4B) models from 0.70 to 0.38.
- **Hallucination persists.** In our hard-case error analysis, recognition errors (literal readings, wrong target/mechanism, missed cultural references) drop by 36–48% with gold labels, but invented context barely changes (31 → 27).
- **Reference-based metrics are weak proxies.** Against human ratings, ROUGE-L, METEOR, SBERT, and BERTScore correlate at ρ = 0.41–0.80, while an LLM judge correlates at **ρ = 0.92**.

---

## Dataset

### Construction

1. **Collection:** Memes were saved by hand from public Facebook, Instagram, and Reddit pages, groups, and communities (March–May 2026). Annotators browsed feeds broadly instead of searching fixed keywords, so the data covers politics, sports, entertainment, trending events, and social issues. Initial pool: 1,421 memes.
2. **Filtering:** We removed duplicates, memes without Bengali text, blurry or unreadable memes, and sexually or visually explicit content, leaving **1,200 memes**. Usernames, profile information, and watermarks were removed.
3. **Text transcription:** Embedded Bengali text was transcribed by hand, because Bengali OCR is still error-prone on stylised meme fonts.
4. **Annotation:** Three native Bengali-speaking annotators labeled every meme independently, supervised by an expert with over ten years of research experience. Labels were decided by majority vote, with the expert deciding unresolved cases. The expert reviewed all explanations and requested revisions where needed.

**Inter-annotator agreement (Fleiss' κ):** 0.77 for target, 0.72 for mechanism.

### Statistics

| Statistic | Value |
|-----------|-------|
| Total memes | 1,200 |
| **Target** | |
| Individual | 679 (56.6%) |
| Concept | 256 (21.3%) |
| Community | 240 (20.0%) |
| Organization | 25 (2.1%) |
| **Mechanism** | |
| Direct Derision | 587 (48.9%) |
| Ironic/Deceptive Framing | 265 (22.1%) |
| Comparative/Stereotypical Ridicule | 235 (19.6%) |
| Provocative Baiting | 113 (9.4%) |
| **Length** | |
| Meme text (words) | 16.2 ± 8.5 |
| Explanation (words) | 73.3 ± 9.0 |

Targets and mechanisms are only moderately associated (Cramér's V = 0.23), so knowing one label reveals little about the other.

### Label Definitions

| Label | Definition |
|-------|-----------|
| **Individual** | A specific real, fictional, or publicly recognisable person |
| **Organization** | A formally established entity, such as a party, company, university, or agency |
| **Community** | People connected by a shared identity, affiliation, occupation, or location |
| **Concept** | A non-human or abstract target, such as a behaviour, policy, trend, or event |
| **Direct Derision** | Openly insulting, humiliating, or ridiculing the target |
| **Comparative/Stereotypical Ridicule** | Degrading the target through an unfavourable comparison, stereotype, or analogy |
| **Ironic/Deceptive Framing** | Attacking indirectly through sarcasm, ironic praise, parody, or text–image contradiction |
| **Provocative Baiting** | Using inflammatory or polarising content to provoke anger or conflict |

---

## Gold-Label Conditioning Protocol

When a model explains a meme poorly, it could be because it failed to *recognise* the target or mechanism, or because it recognised them but failed to *put them into words*. We compare four prompt settings:

| Setting | Given in prompt | Model predicts |
|---------|-----------------|----------------|
| **P0: Plain** | — | Target, mechanism, explanation |
| **P1: + Gold target** | Target | Mechanism, explanation |
| **P2: + Gold mechanism** | Mechanism | Target, explanation |
| **P3: + Both** | Target and mechanism | Explanation |

P3 is a diagnostic setting, not a strict upper bound: models can still misread the meme or invent context.

---

## Models

All models are instruction-tuned and evaluated **zero-shot** (no fine-tuning, no in-context examples).

| Size | Models |
|------|--------|
| Medium (8–12B) | Pixtral-12B, Qwen3.5-9B, InternVL3.5-8B, Llama3-8B |
| Small (≤4B) | Qwen3VL-4B, InternVL3.5-4B, Gemma3-4B, PaliGemma2-3B, SmolVLM2-2.2B, Qwen3VL-2B |

**Inference settings (as in the paper):** meme image plus transcribed text in the prompt; longer image side resized to ≤768 px; greedy decoding; repetition penalty 1.05; up to 320 new tokens; reasoning ("thinking") modes disabled. All experiments were run on Kaggle GPUs.

---

## Evaluation

- **Classification:** per-class F1 (labels are imbalanced), plus accuracy, precision, recall, and confusion matrices.
- **Explanation, automatic:** ROUGE-L, METEOR, BERTScore (RoBERTa-large, rescaled), and SBERT cosine similarity (`all-mpnet-base-v2`).
- **Explanation, LLM judge:** LAVE-style rating (1–5) by Gemini 3 Flash, given the meme, its text, the gold explanation, and the generated explanation.
- **Explanation, human:** 12 native Bengali-speaking raters scored 400 explanations per model and setting (16,000 explanations, 48,000 ratings), blind to model and setting.

---

## Results

### Explanation Quality (Human Rating, 1–5)

| Model | P0 | P1 | P2 | P3 |
|-------|---:|---:|---:|---:|
| Pixtral-12B | 3.38 | 3.23 | 3.31 | 3.43 |
| **Qwen3.5-9B** | **3.45** | **3.56** | **3.58** | **3.67** |
| InternVL3.5-8B | 3.37 | 3.36 | 3.33 | 3.40 |
| Llama3-8B | 3.35 | 3.33 | 3.38 | 3.47 |
| Qwen3VL-4B | 3.25 | 3.18 | 3.24 | 3.36 |
| InternVL3.5-4B | 3.17 | 3.21 | 3.25 | 3.49 |
| Gemma3-4B | 1.63 | 1.71 | 1.87 | 3.01 |
| PaliGemma2-3B | 2.68 | 2.79 | 2.81 | 2.87 |
| SmolVLM2-2.2B | 2.25 | 2.48 | 2.74 | 2.81 |
| Qwen3VL-2B | 3.19 | 3.05 | 3.11 | 3.12 |
| *Mean* | 2.97 | 2.99 | 3.06 | 3.26 |

Mean LLM-judge score rises from 2.90 (P0) to 3.25 (P3). Full automatic-metric results for all settings are in the paper.

### Classification Highlights (Per-Class F1)

- **Target:** Individual is the easiest class (best F1 0.692, InternVL3.5-8B, P2); Organization is the hardest (best 0.383).
- **Mechanism:** Direct Derision is the easiest (best 0.585, Pixtral-12B, P1); Provocative Baiting is the hardest (best only 0.142).
- Giving one gold label improves classification of the other label for every model.

### Error Analysis (100 Hard Memes per Setting)

| Error | Qwen3.5-9B P0 → P3 | Gemma3-4B P0 → P3 |
|-------|:---:|:---:|
| Ignores meme text | 1 → 0 | 31 → 10 |
| Literal reading | 24 → 13 | 40 → 23 |
| Wrong target | 21 → 11 | 29 → 18 |
| Wrong mechanism | 25 → 16 | 33 → 21 |
| **Hallucinated context** | **31 → 27** | 37 → 23 |
| Missed cultural reference | 20 → 11 | 31 → 19 |
| Repeats label only | 3 → 1 | 13 → 7 |

---

## Repository Structure

```
BanTrollEx/
├── notebooks/
│   ├── P0_plain_prompting.ipynb      # predicts target + mechanism + explanation
│   ├── P1_gold_target.ipynb          # gold target given; predicts mechanism + explanation
│   ├── P2_gold_mechanism.ipynb       # gold mechanism given; predicts target + explanation
│   └── P3_gold_both.ipynb            # both gold labels given; predicts explanation
├── evaluation/
│   ├── classification_metrics.py     # per-class F1, accuracy, confusion matrices
│   ├── explanation_metrics.py        # ROUGE-L, METEOR, BERTScore, SBERT
│   └── llm_judge.py                  # LAVE-style LLM-as-a-judge
├── prompts/                          # prompt templates for P0–P3 and the judge
├── annotation_tool/                  # Chihna, our offline Bengali meme annotation tool
├── data/                             # annotations (see Data Access)
├── requirements.txt
└── README.md
```

Each notebook includes a robust JSON output parser, automatic metric calculation, class-wise metrics, confusion matrices, and LaTeX table export.

---

## Getting Started

### Run on Kaggle (recommended)

1. Create a Kaggle notebook and upload one of the files in `notebooks/`.
2. Settings: **Accelerator → GPU T4 ×2**, **Internet → On**.
3. Add the BanTrollEx dataset as input and set `DATA_FILE` and `IMAGE_ROOT` in the configuration cell.
4. Set `MODEL_ID` (e.g., `Qwen/Qwen3.5-9B`) and `LOAD_MODE` (`fp16` across two T4s, or `4bit` for a single GPU).
5. Run all cells. Predictions, metrics, confusion matrices, and LaTeX tables are saved to `/kaggle/working/results/<RUN_NAME>/`.

Use `LIMIT = 20` for a quick test before running the full dataset.

### Run locally

```bash
git clone https://github.com/<your-username>/BanTrollEx.git
cd BanTrollEx
pip install -r requirements.txt
```

### Data format

The notebooks accept the JSON exported by Chihna or a CSV with these columns:

| Column | Description |
|--------|-------------|
| `image` | Image filename or relative path |
| `target_category` | Gold target label |
| `trolling_mechanism` | Gold mechanism label |
| `explanation` | Gold explanation |

---

## Chihna Annotation Tool

**Chihna** (Bengali for "mark" or "sign") is a lightweight, browser-based annotation tool we built for Bengali memes. It runs **entirely offline** from a local web server, so images and annotations never leave the annotator's machine. It supports troll status, target, mechanism, and explanation fields; validation before saving; skipping uncertain memes for expert review; auto-saved drafts; resumable checkpoints; and JSON export/import for merging annotators' work.

---

## Data Access

- **Annotations** (target, mechanism, explanation) are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Meme images** are available for non-commercial research only, under a data-use agreement that prohibits redistribution, re-identification of individuals, harassment, and the creation of troll content. To request access, email the contact below.

---

## Ethical Considerations

All memes were collected by hand from publicly accessible pages without interacting with users, and identifying metadata was removed. Public figures targeted by the memes may still be recognisable; including them does not mean we endorse the memes' claims. Annotators were informed about offensive content in advance, took part voluntarily, could stop at any time, and were paid according to local standards. Labels reflect a documented annotation perspective, not an objective ground truth. Because current VLMs often misidentify targets and mechanisms and may reproduce stereotypes, models evaluated on BanTrollEx should not be used for automated moderation without human oversight.

---

## Citation

The paper is in preparation. Until it is published, please cite this repository:

```bibtex
@misc{shanto2026bantrollex,
  title  = {BanTrollEx: Who, How, and Why? Explaining Bengali Troll Memes with Vision-Language Models},
  author = {Shanto, Anik Mahmud and others},
  year   = {2026},
  url    = {https://github.com/<your-username>/BanTrollEx}
}
```

## Contact

**Anik Mahmud Shanto**, Lecturer, Department of Computer Science and Engineering, Southeast University, Bangladesh
Email: anik.mahmudshanto@seu.edu.bd
