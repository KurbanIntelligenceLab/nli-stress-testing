# Multilingual Logical Inference with LLMs

This repository contains the codebase and dataset artifacts for our paper:

**“Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference”**

We introduce a controlled framework for evaluating the multilingual reasoning capabilities of large language models (LLMs) using synthetic natural language inference (NLI) stimuli, high-quality translation pipelines, and code-switching scenarios. Our method probes cross-lingual alignment and logical consistency by stress-testing models across diverse language pairs, sentence perturbations, and embedding visualizations.

## 🔍 Project Overview

- ✅ **Synthetic NLI Generation** — Custom templates encoding entailment, contradiction, and neutrality
- 🌍 **Multilingual Translation** — Automated translation into 5+ diverse languages (incl. Hindi, Swahili, Arabic)
- 🔄 **Code-Switching Probes** — Mixed-language premise/hypothesis evaluation to test multilingual robustness
- 🧠 **LLM Inference Classification** — Prompt-based prediction pipeline for zero-shot NLI across languages
- 📈 **Analysis & Visualization** — Accuracy reporting and analysis


## 🛠️ How to Run


1. Clone the repo and install required packages:

```bash
git clone https://github.com/KurbanIntelligenceLab/nli-stress-testing.git
cd nli-stress-testing
```

2. Ensure that all required LLMs (open-source and gated models) are accessible through huggingface, and login with your huggingface token:

```bash
huggingface-cli login
```

3. Run **monolingual** NLI analysis using generated data available under 'multilingual_nli' repository:
```bash
python llm_running.py
```

4. Run **code-switched** NLI analysis using generated data available under multilingual_nli repository:
```bash
python llm_running_code_switching.py
```
