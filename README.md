# DyLM-OHCA

> **Deep Language Model-based Early Recognition of Out-of-Hospital Cardiac Arrest from Real-Time Emergency Calls**
>
> *npj Digital Medicine (2026)* · [Paper Link](https://doi.org/10.1038/s41746-026-02498-5)

---

## 📋 Overview

**DyLM-OHCA** is a dynamic deep language model designed to recognize out-of-hospital cardiac arrest (OHCA) in **real time within the first 60 seconds** of emergency calls. Built on a GPT-2 decoder backbone with causal attention, it processes the evolving dispatcher–caller dialogue incrementally — moving beyond keyword spotting to capture full conversational context.

<div align="center">

| Metric | Value |
|--------|-------|
| **AUROC** | 0.937 |
| **AUPRC** | 0.456 |
| **Median Recognition Time** | 19.7 s |
| **Specificity @ Recall 0.80** | 92.4% |

</div>

---

## 🔍 Why It Matters

Every minute of CPR delay reduces survival by **7–10%**. Yet even experienced emergency dispatchers fail to identify OHCA in **20–30% of calls**. DyLM-OHCA provides:

- ⚡ **Fast alerts** — 75% of OHCA-positive calls identified within the first 29.1 s
- 🧠 **Context-aware predictions** — understands conversational flow, not just keywords
- 🔎 **Interpretable outputs** — word-level leave-one-out attribution for dispatcher trust
- 📈 **Dynamic risk scores** — real-time trajectory that can self-correct over the call

---

## 🏗️ Model Architecture

```
Emergency Call Transcript (Korean)
        ↓
   Tokenizer (GPT-2)
        ↓
 Decoder (GPT-2 Backbone)
   [Causal Attention]
        ↓
  Token-level Classifier
   [3-class: OHCA / Severe non-OHCA / Non-severe]
        ↓
  Step 1: Sentence-level averaging
        ↓
  Step 2: Moving average (K=3)
        ↓
  Threshold Crossing → 🚨 OHCA Alert
```

The model is trained end-to-end with **weakly supervised learning** using only call-level labels, yet produces token-level risk scores as the conversation unfolds.

---

## 📊 Performance

### Model Comparison (60-second window, Recall ≈ 0.80)

| Model | AUROC | AUPRC | Precision | Specificity |
|-------|-------|-------|-----------|-------------|
| Logistic Regression | 0.865 | 0.213 | 0.065 | 0.808 |
| XGBoost | 0.882 | 0.280 | 0.060 | 0.791 |
| Gradient Boosting | 0.847 | 0.082 | 0.051 | 0.753 |
| Random Forest | 0.838 | 0.161 | 0.042 | 0.699 |
| **DyLM-OHCA** | **0.937** | **0.456** | **0.155** | **0.924** |

### Recognition Time Comparison vs. Prior Work

| Study | Median Recognition Time | 60-s Recognition Rate |
|-------|------------------------|----------------------|
| Blomberg et al. (2019) | 44 s | — |
| Byrsell et al. (2021) | 52–85 s | ~27–50% |
| **DyLM-OHCA** | **19.7 s** | **≥75%** |

---

## 🔑 Key Findings

### Linguistic Patterns Driving Recognition

**True Positive — Caller words:** `looks like/think` · `no` · `don't/not` · `breathing`

**True Positive — Dispatcher words:** `conscious` · `don't/not` · `breath` · `is breathing` · `address`

> OHCA recognition was driven more by **conversational flow** than by individual keywords. A typical trigger: the dispatcher asks *"Is the patient conscious?"* and the caller responds *"No, he's not waking up."*

### Prediction Stability (TP vs. FP)

| | Sustained Score | Drop-off Score |
|-|----------------|---------------|
| **True Positive** | 93.2% | 6.8% |
| **False Positive** | 47.4% | **52.6%** |

Over half of false positives show risk score *decline* after the initial alert — a useful real-world cue for dispatchers to reevaluate.

---

## 🗂️ Dataset

- **Source:** [AI Hub](https://www.aihub.or.kr/) — 119 Intelligent Emergency Call Voice Recognition Dataset
- **Size:** 158,973 emergency call transcripts (3,064 hours of audio)
- **Regions:** Seoul (n=104,496), Incheon (n=28,428), Gwangju (n=26,049)
- **Year:** 2023
- **OHCA prevalence:** 1.7% (n=2,646)
- **Split:** Train 80% / Test 20% (stratified)

> ⚠️ The dataset requires registration and approval on the AI Hub platform. It is not directly redistributed in this repository.

---

## ⚙️ Installation

```bash
git clone https://github.com/ggomaeng514/DyLM-OHCA.git
cd DyLM-OHCA
pip install -r requirements.txt
```

---

## 📝 Citation

```bibtex
@article{choi2026deep,
  title={Deep language model-based early recognition of out-of-hospital cardiac arrest from real-time emergency calls},
  author={Choi, Hong-Jae and Hwang, Minyoung and Cho, Sangyeon and Kim, Hyunsoo and Kim, Junyeong and Bae, Woori and Lee, Changhee},
  journal={npj Digital Medicine},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

---

## 👥 Authors

**Hong-Jae Choi**¹†, **Minyoung Hwang**²†, **Sangyeon Cho**³˒⁵, **Hyunsoo Kim**⁴, **Junyeong Kim**³, **Woori Bae**⁶, **Changhee Lee**²*

¹ SMG-SNU Boramae Medical Center, Seoul National University · ² Korea University · ³ Chung-Ang University · ⁴ Incheon Fire Headquarters · ⁵ Korea Surgical Research Foundation · ⁶ Seoul St. Mary's Hospital, The Catholic University of Korea

†Equal contribution · *Correspondence: changheelee@korea.ac.kr

---

## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).
