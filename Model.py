#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WavLM 5-class Emotion (Relabeled) Classification on AbstractTTS/IEMOCAP.

Relabel scheme (5 classes):
- high_neg: angry + frustrated  
- excited:  excited            
- happy:    happy
- neutral:  neutral
- sad:      sad
"""

import sys
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset, Audio, Dataset, disable_caching
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# 0. Config & Seed
# --------------------
ORIG_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]

DATASET_NAME = "AbstractTTS/IEMOCAP"
LABEL_COL = "major_emotion"

MODEL_NAME = "microsoft/wavlm-base"

MAX_DURATION_SECONDS = 6.0
TEST_SIZE = 0.1
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# 1. Load & Filter Dataset
# --------------------
def load_and_filter_dataset():
    disable_caching()
    print("Python:", sys.version)
    print("Using device:", device)

    print("Loading dataset...", DATASET_NAME)
    ds_raw = load_dataset(DATASET_NAME)
    full_ds: Dataset = ds_raw["train"]

    full_ds = full_ds.cast_column("audio", Audio(decode=False))
    print("Columns:", full_ds.column_names)

    keep_set = set(ORIG_LABELS)

    def keep_six(example):
        return example[LABEL_COL] in keep_set

    filtered = full_ds.filter(keep_six)
    print("Filtered size:", len(filtered))
    print("Original label distribution:", Counter(filtered[LABEL_COL]))

    all_labels = np.array(filtered[LABEL_COL])
    all_indices = np.arange(len(filtered))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=all_labels,
    )

    train_ds = filtered.select(train_idx.tolist())
    val_ds = filtered.select(val_idx.tolist())

    print("Train size:", len(train_ds), "Val size:", len(val_ds))
    print("Train label dist:", Counter(train_ds[LABEL_COL]))
    print("Val   label dist:", Counter(val_ds[LABEL_COL]))

    return train_ds, val_ds


# --------------------
# 2. Relabel to 5-class scheme
# --------------------
NEW_LABELS = ["high_neg", "excited", "happy", "neutral", "sad"]

EMO2NEW = {
    "angry": "high_neg",
    "frustrated": "high_neg",
    "excited": "excited",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}


def add_relabel(train_ds: Dataset, val_ds: Dataset):
    # new label name -> id
    new2id = {name: i for i, name in enumerate(NEW_LABELS)}
    id2new = {i: name for name, i in new2id.items()}
    num_labels = len(new2id)
    print("new label2id:", new2id)

    def _add_new(example):
        emo = example[LABEL_COL]
        new_name = EMO2NEW[emo]
        example["relabel"] = new_name
        example["label_id"] = new2id[new_name]
        return example

    train_ds = train_ds.map(_add_new)
    val_ds = val_ds.map(_add_new)

    print("Train relabel dist:", Counter(train_ds["relabel"]))
    print("Val   relabel dist:", Counter(val_ds["relabel"]))
    print("Train label_id dist:", Counter(train_ds["label_id"]))
    print("Val   label_id dist:", Counter(val_ds["label_id"]))

    return train_ds, val_ds, new2id, id2new, num_labels


# --------------------
# 3. Decode Audio to Waveform
# --------------------
def decode_audio_columns(train_ds: Dataset, val_ds: Dataset):
    def decode_audio(example):
        import soundfile as sf
        import io
        import numpy as np

        raw_bytes = example["audio"]["bytes"]
        waveform, sr = sf.read(io.BytesIO(raw_bytes))
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # stereo -> mono
        example["waveform"] = waveform
        example["orig_sr"] = sr
        return example

    train_ds = train_ds.map(decode_audio, desc="Decoding train audio")
    val_ds = val_ds.map(decode_audio, desc="Decoding val audio")

    return train_ds, val_ds


# --------------------
# 4. Build FeatureExtractor & Dataloaders
# --------------------
def build_dataloaders(train_ds: Dataset, val_ds: Dataset):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    target_sr = feature_extractor.sampling_rate
    print("Target SR:", target_sr)

    max_len_samples = int(MAX_DURATION_SECONDS * target_sr)

    def collate_fn(batch):
        import librosa

        waveforms = []
        labels = []
        for ex in batch:
            waveform = ex["waveform"]
            sr = ex["orig_sr"]
            if sr != target_sr:
                waveform = librosa.resample(
                    waveform, orig_sr=sr, target_sr=target_sr
                )
            if len(waveform) > max_len_samples:
                waveform = waveform[:max_len_samples]
            waveforms.append(waveform)
            labels.append(ex["label_id"])

        inputs = feature_extractor(
            waveforms,
            sampling_rate=target_sr,
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    batch = next(iter(train_loader))
    print("Example batch shapes:", {k: v.shape for k, v in batch.items()})

    return feature_extractor, train_loader, val_loader


# --------------------
# 5. Plotting helpers
# --------------------
def plot_confusion_matrix(cm, labels, title="Confusion Matrix (val)", save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_confusion_matrix] Saved to {save_path}")
    plt.close()


def plot_loss_curves(history, save_path=None):
    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train"], label="Train loss")
    plt.plot(epochs, history["val"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_loss_curves] Saved to {save_path}")
    plt.close()


# --------------------
# 6. Model with Attentive Pooling (5-way)
# --------------------
class AttentiveStatsPooling(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):

        B, T, H = hidden_states.size()
        mask = torch.ones(B, T, device=hidden_states.device, dtype=torch.long)

        attn_logits = self.attention(hidden_states)  # [B, T, 1]
        mask_ = mask.unsqueeze(-1)                   # [B, T, 1]
        attn_logits = attn_logits.masked_fill(mask_ == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, T, 1]

        mean = torch.sum(attn_weights * hidden_states, dim=1)  # [B, H]
        diff = hidden_states - mean.unsqueeze(1)               # [B, T, H]
        var = torch.sum(attn_weights * diff * diff, dim=1)    # [B, H]
        std = torch.sqrt(torch.clamp(var, min=1e-9))          # [B, H]

        stats = torch.cat([mean, std], dim=-1)  # [B, 2H]
        return stats


class WavLMRelabel5Classifier(nn.Module):

    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.wavlm.config.hidden_size

        self.pooling = AttentiveStatsPooling(hidden_size)
        pooled_dim = hidden_size * 2

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(pooled_dim, num_labels)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = self.pooling(hidden_states)       # [B, 2H]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def build_model_and_optim(train_ds: Dataset, num_labels: int, train_loader):
    model = WavLMRelabel5Classifier(
        base_model_name=MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    train_labels = np.array(train_ds["label_id"])
    counts = np.bincount(train_labels, minlength=num_labels).astype(np.float32)
    class_weights = (len(train_labels) / (num_labels * counts))
    print("Class counts:", counts)
    print("Class weights:", class_weights)

    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = len(train_loader)
    t_total = NUM_EPOCHS * num_update_steps_per_epoch
    warmup_steps = int(0.1 * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
    )
    print("Total training steps:", t_total, "Warmup steps:", warmup_steps)

    return model, criterion, optimizer, scheduler


# --------------------
# 7. Training Loop
# --------------------
def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    id2label,
):
    best_val_acc = 0.0
    best_model_path = "best_wavlm_relabel5.pt"
    num_labels = len(id2label)

    history = {"train": [], "val": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            logits = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
            )
            labels = batch["labels"]
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        history["train"].append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"],
                )
                labels = batch["labels"]

                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        history["val"].append(avg_val_loss)
        val_acc = correct / total if total > 0 else 0.0

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print(
            f"\nEpoch {epoch+1}/{NUM_EPOCHS} finished | "
            f"train_loss = {avg_train_loss:.4f} | "
            f"val_loss = {avg_val_loss:.4f} | "
            f"val_acc = {val_acc:.4f}"
        )

        target_names = [id2label[i] for i in range(num_labels)]
        print("\nClassification report (relabel 5-class):")
        print(
            classification_report(
                all_labels, all_preds, target_names=target_names, digits=4
            )
        )

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("Index -> Label:", {i: id2label[i] for i in range(num_labels)})
        print(cm)
        print("\n" + "=" * 80 + "\n")

        plot_confusion_matrix(
            cm,
            labels=target_names,
            title=f"Relabel 5-class Confusion Matrix (val, epoch {epoch+1})",
            save_path=f"cm_relabel5_epoch_{epoch+1}.png",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "id2label": id2label,
                    "label2id": {v: k for k, v in id2label.items()},
                    "new_labels": NEW_LABELS,
                    "model_name": MODEL_NAME,
                },
                best_model_path,
            )
            print(
                f"✨ New best model saved (val_acc = {val_acc:.4f}) "
                f"-> {best_model_path}"
            )

    print(f"\nTraining finished. Best val_acc (relabel-5) = {best_val_acc:.4f}")
    plot_loss_curves(history, save_path="loss_curves_relabel5.png")
    print("Loss curves saved to loss_curves_relabel5.png")


# --------------------
# 8. Main
# --------------------
def main():
    set_seed(RANDOM_SEED)

    train_ds, val_ds = load_and_filter_dataset()
    train_ds, val_ds, label2id, id2label, num_labels = add_relabel(train_ds, val_ds)
    train_ds, val_ds = decode_audio_columns(train_ds, val_ds)
    feature_extractor, train_loader, val_loader = build_dataloaders(train_ds, val_ds)
    model, criterion, optimizer, scheduler = build_model_and_optim(
        train_ds, num_labels, train_loader
    )
    train(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        id2label,
    )


if __name__ == "__main__":
    main()
