import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import evaluate
import wandb

#CONFIG 
model_name = "google/paligemma-3b-pt-224"
csv_path = "captions.csv"
image_dir = "resized"
output_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)

#LOAD DATA
df = pd.read_csv(csv_path)
df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, x))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
test_df = df[df["split"] == "test"]

# LOAD MODEL 
processor = AutoProcessor.from_pretrained(model_name)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)
model.eval()


#INFERENCE 
results = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="🖼️ Running inference"):
    try:
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(text="<image> caption en:", images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=6,
                do_sample=False,
                early_stopping=True
            )

        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        results.append({"image": row["image"], "generated_caption": caption})
        print(f"{row['image']} → {caption}")

    except Exception as e:
        print(f"[ERROR] {row['image']}: {e}")
        results.append({"image": row["image"], "generated_caption": ""})


pred_path = os.path.join(output_dir, "predictions.csv")
pd.DataFrame(results).to_csv(pred_path, index=False)
print(f"Captions saved to {pred_path}")


#EVALUATION
wandb.init(project="paligemma-captioning", name="eval-metrics")

df_pred = pd.read_csv(pred_path)
df_gt = pd.read_csv(csv_path)
df_gt = df_gt[df_gt["image"].isin(df_pred["image"])]

df = df_gt.merge(df_pred, on="image")
caption_cols = [f"caption_{i}" for i in range(1, 6)]
df[caption_cols] = df[caption_cols].fillna("").astype(str)

predictions = df["generated_caption"].fillna("").astype(str).tolist()
multi_references = df[caption_cols].values.tolist()


# Load metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# BLEU
bleu_result = bleu.compute(predictions=predictions, references=multi_references)
wandb.log({
    "BLEU/score": bleu_result["bleu"],
    "BLEU/precision_1gram": bleu_result["precisions"][0],
    "BLEU/precision_2gram": bleu_result["precisions"][1],
    "BLEU/precision_3gram": bleu_result["precisions"][2],
    "BLEU/precision_4gram": bleu_result["precisions"][3]
})

# METEOR
meteor_result = meteor.compute(predictions=predictions, references=multi_references)
wandb.log({"METEOR": meteor_result["meteor"]})

# ROUGE (averaged over each caption)
rouge_scores = []
for col in caption_cols:
    scores = rouge.compute(predictions=predictions, references=df[col].tolist())
    rouge_scores.append(scores)

avg_rouge = {k: np.mean([score[k] for score in rouge_scores]) for k in rouge_scores[0].keys()}

wandb.log({
    "ROUGE/rouge1": avg_rouge["rouge1"],
    "ROUGE/rouge2": avg_rouge["rouge2"],
    "ROUGE/rougeL": avg_rouge["rougeL"],
    "ROUGE/rougeLsum": avg_rouge["rougeLsum"]
})




print("\n Final Evaluation Metrics:")
print("BLEU:", bleu_result)
print("METEOR:", meteor_result)
print("Averaged ROUGE over 5 captions:", avg_rouge)

wandb.finish()
