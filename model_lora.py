import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from evaluate import load
import wandb
from IPython.display import display

# CONFIG
model_name = "google/paligemma-3b-pt-224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_epochs = 1
max_length = 128


model_name = "google/paligemma-3b-pt-224"
csv_path = "captions.csv"                     
image_dir = "resized"                       
save_dir = "checkpoints"   

# INIT WANDB
wandb.init(
    project="paligemma-lora-captioning",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "prompt": "caption en: <image>",
        "unfreeze_backbone": True,
        "learning_rate": 1e-5
    }
)

# LOAD DATA
df = pd.read_csv(csv_path)
df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, x))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

train_raw = df[df["split"] == "train"].sample(n=4000, random_state=42)
expanded_rows = []

for _, row in train_raw.iterrows():
    caption_indices = [i for i in range(1, 6) if pd.notna(row[f"caption_{i}"])]
    selected_indices = caption_indices[:3]
    for i in selected_indices:
        new_row = row.copy()
        new_row["caption"] = row[f"caption_{i}"]
        expanded_rows.append(new_row)

train_df = pd.DataFrame(expanded_rows).reset_index(drop=True)

val_df = df[df["split"] == "val"]
val_df["caption"] = val_df["caption_1"]

test_df = df[df["split"] == "test"]
test_df["caption"] = test_df["caption_1"]

wandb.config.update({
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "using_3_captions_per_image": True
})

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# DATASET
class RISCDataset(Dataset):
    def __init__(self, df, processor, max_length=128):
        self.data = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        prompt = "caption en: <image>"
        caption = row["caption"]

        # Use suffix=caption (like the notebook)
        inputs = self.processor(
            images=image,
            text=prompt,
            suffix=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


# LOAD MODEL WITH LoRA
print(" Loading base model with LoRA...")

from peft import LoraConfig, get_peft_model

# Load processor
processor = AutoProcessor.from_pretrained(model_name)

# Define LoRA configuration (same as notebook)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Load the base model
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)

# Apply LoRA adapters
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# TRAINING
train_dataset = RISCDataset(train_df, processor, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        wandb.log({"train_loss": loss.item(), "step": epoch * len(train_loader) + step})

    avg_loss = total_loss / len(train_loader)
    print(f" Epoch {epoch+1} Loss: {avg_loss:.4f}")
    wandb.log({"avg_epoch_loss": avg_loss, "epoch": epoch+1})

# SAVE LOCALLY
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Model + LoRA saved locally to: {save_dir}")


# EVALUATION
bleu = load("bleu")
meteor = load("meteor")
rouge = load("rouge")

def evaluate_split(split_df, split_name):
    predictions, references = [], []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Evaluating {split_name}"):
        image = Image.open(row["image_path"]).convert("RGB")
        refs = [row[f"caption_{i}"] for i in range(1, 6) if pd.notna(row[f"caption_{i}"])]
        refs = refs[:3]

        inputs = processor(
            images=image,
            text="caption en: <image>",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4)
            pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        predictions.append(pred)
        references.append(refs)

    bleu_score = bleu.compute(predictions=predictions, references=references, max_order=1)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)["rougeL"]

    wandb.log({
        f"{split_name}_bleu": bleu_score["bleu"],
        f"{split_name}_meteor": meteor_score["meteor"],
        f"{split_name}_rougeL": rouge_score
    })

    print(f"\n Final {split_name.upper()} Metrics:")
    print(f"BLEU-1:   {bleu_score['bleu']:.4f}")
    print(f"METEOR:   {meteor_score['meteor']:.4f}")
    print(f"ROUGE-L:  {rouge_score:.4f}")

# RUN EVALUATION
evaluate_split(val_df, "val")
evaluate_split(test_df, "test")

import matplotlib.pyplot as plt

def visualize_predictions(split_df, split_name, num_samples=2):
    model.eval()
    samples = split_df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    for idx, row in samples.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        refs = [row[f"caption_{i}"] for i in range(1, 6) if pd.notna(row[f"caption_{i}"])]
        refs = refs[:3]  # Limit to 3 refs

        # Generate caption
        inputs = processor(
            images=image,
            text="caption en: <image>",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4)
            pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        # Show image and captions
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"[{split_name.upper()} Sample {idx+1}]\nPredicted: {pred}\nRef: {refs[0]}")
        plt.show()

#  Show 2 val and test samples
visualize_predictions(val_df, "val", num_samples=2)
visualize_predictions(test_df, "test", num_samples=2)

print("\n Showing test samples with predicted captions\n")
num_samples = min(3, len(test_df))
sample_df = test_df.sample(n=num_samples, random_state=42).reset_index(drop=True)

table = wandb.Table(columns=["Image", "Prediction", "Reference_1", "Reference_2", "Reference_3"])

for idx, row in sample_df.iterrows():
    image = Image.open(row["image_path"]).convert("RGB")
    display(image)
    references = [row[f"caption_{i}"] for i in range(1, 6) if pd.notna(row[f"caption_{i}"])]
    references = references[:3]

    inputs = processor(
        images=image,
        text="caption en: <image>",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        pred_caption = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    print(f"\n Image: {row['image']}")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    print(f"\n Predicted Caption:\n  {pred_caption}")
    print("-" * 60)

    table.add_data(wandb.Image(image), pred_caption, *references)

wandb.log({"sample_captions": table})


# FINISH
wandb.finish()



