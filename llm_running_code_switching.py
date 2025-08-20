!huggingface-cli login

import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check CUDA
assert torch.cuda.is_available(), "🚨 CUDA is not available!"
device = torch.device("cuda")

# Load model + tokenizer

models = ["meta-llama/Meta-Llama-3-8B-Instruct",'mistralai/Mistral-7B-Instruct-v0.3','QCRI/Fanar-1-9B-Instruct','google/gemma-7b-it',"microsoft/phi-4","Qwen/Qwen3-1.7B"]
for MODEL_NAME in models:
# MODEL_NAME = "Qwen/Qwen3-1.7B"
  model_n = MODEL_NAME.split("/")[-1]
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      device_map="auto",
      torch_dtype=torch.float16
  )
  model.eval()

  # Confirm GPU usage
  print("✅ Using device:", next(model.parameters()).device)

  # Directory setup
  DATA_DIR = "code_switch/data"
  OUTPUT_DIR = "code_switch/output"
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # Define choices
  CHOICES = ["entailment", "contradiction", "neutral"]
  if MODEL_NAME == 'microsoft/phi-4' or MODEL_NAME == 'Qwen/Qwen3-1.7B' :
    CHOICE_TOKENS = {
        c: tokenizer(c, return_tensors="pt").input_ids[0][0].item()
        for c in CHOICES
    }

  else:
    CHOICE_TOKENS = {
        c: tokenizer(c, return_tensors="pt").input_ids[0][1].item()
        for c in CHOICES
    }

  # Prompt template
  def build_prompt(premise, hypothesis):
      return (
          f"Premise: {premise}\n"
          f"Hypothesis: {hypothesis}\n"
          f"Question: Is the hypothesis entailed by the premise, contradicted by it, or unrelated?\n"
          f"Answer with one of: Entailment, Contradiction, Neutral.\n"
          f"Answer:"
      )

  # Batched classification
  def classify_batch(prompts, batch_size=8):
      inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
      with torch.no_grad():
          outputs = model(**inputs)
      logits = outputs.logits[:, -1, :]
      probs = torch.nn.functional.softmax(
          logits[:, [CHOICE_TOKENS[k] for k in CHOICES]], dim=1
      )
      preds = torch.argmax(probs, dim=1).tolist()
      return [CHOICES[p] for p in preds]

  # Process each language file
  for file in os.listdir(DATA_DIR):
      if not file.endswith(".json"):
          continue

      lang = file.split("/")[-1]
      print(f"\n🌍 Processing language: {lang}")

      input_path = os.path.join(DATA_DIR, file)
      output_path = os.path.join(OUTPUT_DIR, f"{model_n}_{lang}.json")

      with open(input_path, "r", encoding="utf-8") as f:
          examples = [json.loads(line) for line in f]

      outputs = []
      BATCH_SIZE = 8
      for i in tqdm(range(0, len(examples), BATCH_SIZE), desc=f"Classifying {lang}"):
          batch = examples[i:i + BATCH_SIZE]
          prompts = [build_prompt(ex["premise"], ex["hypothesis"]) for ex in batch]
          try:
              preds = classify_batch(prompts)
              for ex, pred in zip(batch, preds):
                  ex["prediction"] = pred
                  outputs.append(ex)
          except Exception as e:
              print(f"⚠️ Error in batch: {e}")
              continue

      with open(output_path, "w", encoding="utf-8") as f:
          for ex in outputs:
              json.dump(ex, f, ensure_ascii=False)
              f.write("\n")

  print("\n✅ Done! All results saved.")
  

import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory where classified files are saved
OUTPUT_DIR = "code_switch/output"

# Collect results
results = []
for file in os.listdir(OUTPUT_DIR):
    if not file.endswith(".json"):
        continue

    lang = file
    model_n=file.split("_")[0].split(".")[0]
    with open(os.path.join(OUTPUT_DIR, file), "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    y_true = [ex["label"].strip().lower() for ex in data]
    y_pred = [ex["prediction"].strip().lower() for ex in data]

    acc = accuracy_score(y_true, y_pred)
    results.append((lang, acc))
    print('----------------------')

    print(f"\n Language: {lang.split('_')[1:]}, Model: {model_n}")

    print("Accuracy:", round(acc, 4))
    # print("Classification Report:")
    # print(classification_report(y_true, y_pred, labels=["entailment", "contradiction", "neutral"], zero_division=0))
    
    # # Optional: Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred, labels=["entailment", "contradiction", "neutral"])
    # df_cm = pd.DataFrame(cm, index=["entailment", "contradiction", "neutral"], columns=["entailment", "contradiction", "neutral"])
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"Confusion Matrix: {lang}")
    # plt.ylabel("Actual")
    # plt.xlabel("Predicted")
    # plt.tight_layout()
    # plt.show()

# # Show overall accuracy summary
# df_results = pd.DataFrame(results, columns=["Language", "Accuracy"]).sort_values("Accuracy", ascending=False)
# df_results.reset_index(drop=True, inplace=True)