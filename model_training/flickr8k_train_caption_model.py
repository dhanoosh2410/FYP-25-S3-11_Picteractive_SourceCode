"""
flickr8k_train_caption_model.py

Standalone script to:

1. PREPARE captions.txt from the original Flickr8k token file.
2. TRAIN a simple CNN+LSTM image captioning model on Flickr8k (PyTorch).
3. EVALUATE the model using BLEU-4 and save a training/accuracy graph.

This script is **separate** from your FastAPI app.
You run it offline to show how the model is trained and how accuracy is measured.

Usage examples:

# 1) Prepare captions.txt from raw Flickr8k.token
python flickr8k_train_caption_model.py --mode prepare \
    --raw_tokens server/data/Flickr8k.token \
    --out_captions server/data/captions.txt

# 2) Train captioning model and log metrics
python flickr8k_train_caption_model.py --mode train \
    --images_dir server/data/Images \
    --captions_file server/data/captions.txt \
    --epochs 10

This will save:
- models/flickr8k_cnn_lstm.pt          (trained weights)
- models/flickr8k_vocab.json           (vocabulary)
- models/flickr8k_history.json         (loss + BLEU per epoch)
- models/flickr8k_training_curves.png  (graph of loss and BLEU)

Requirements (in addition to your existing ones):
- torchvision
- matplotlib
- nltk  (already in your requirements.txt)

Make sure you've downloaded NLTK punkt tokenizer at least once:
>>> import nltk; nltk.download("punkt")
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# Ensure tokenizer available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ----------------------------
# 1. Utility: prepare captions.txt
# ----------------------------

def prepare_captions_file(raw_tokens_path: Path, out_path: Path) -> None:
    """
    Convert the original Flickr8k.token file format:

        image.jpg#0\tA boy is playing with a dog.
        image.jpg#1\tThe child is running with the dog.
        ...

    into a simpler tab-separated file used by your app:

        image.jpg\tA boy is playing with a dog.
        image.jpg\tThe child is running with the dog.

    This matches the format expected by _read_flickr8k_tokens()
    in showtellpyTorch.py.
    """
    raw_tokens_path = Path(raw_tokens_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[str, str]] = []
    with raw_tokens_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                img_id, cap = line.split("\t", 1)
            else:
                # Fallback: split on first space
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                img_id, cap = parts
            img_name = img_id.split("#", 1)[0]
            pairs.append((img_name, cap))

    with out_path.open("w", encoding="utf-8") as out:
        for img_name, cap in pairs:
            out.write(f"{img_name}\t{cap}\n")

    print(f"[prepare] Wrote {len(pairs)} caption lines to {out_path}")


# ----------------------------
# 2. Dataset + vocabulary
# ----------------------------

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "<unk>": 3,
}


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.word2idx: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2word: Dict[int, str] = {i: w for w, i in SPECIAL_TOKENS.items()}
        self.word_freq: Dict[str, int] = {}
        self.min_freq = min_freq

    def build_from_captions(self, captions: List[str]) -> None:
        for cap in captions:
            for w in nltk.word_tokenize(cap.lower()):
                self.word_freq[w] = self.word_freq.get(w, 0) + 1

        for w, freq in self.word_freq.items():
            if freq >= self.min_freq and w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def encode(self, text: str) -> List[int]:
        tokens = [self.word2idx["<start>"]]
        for w in nltk.word_tokenize(text.lower()):
            tokens.append(self.word2idx.get(w, self.word2idx["<unk>"]))
        tokens.append(self.word2idx["<end>"])
        return tokens

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.word2idx["<end>"]:
                break
            if i in (self.word2idx["<pad>"], self.word2idx["<start>"]):
                continue
            words.append(self.idx2word.get(i, "<unk>"))
        return " ".join(words)

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<pad>"]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "min_freq": self.min_freq,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab


def read_captions_file(captions_file: Path) -> Dict[str, List[str]]:
    """
    Read the same captions.txt format your CLIP retrieval captioner uses.

    Supported formats:
    - TSV: image.jpg\tcaption
    - CSV: image.jpg,caption (header 'image,caption' is skipped)
    """
    captions_file = Path(captions_file)
    caps: Dict[str, List[str]] = {}

    with captions_file.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)

        if "\t" in first:
            # TSV
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img, cap = line.split("\t", 1)
                img = img.split("#", 1)[0].strip()
                cap = cap.strip()
                if img and cap:
                    caps.setdefault(img, []).append(cap)
        else:
            # CSV
            import csv

            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if row[0].strip().lower() == "image" and row[1].strip().lower() == "caption":
                    continue
                img = row[0].split("#", 1)[0].strip()
                cap = row[1].strip()
                if img and cap:
                    caps.setdefault(img, []).append(cap)
    return caps


class Flickr8kCaptionDataset(Dataset):
    def __init__(self, images_dir: Path, captions_file: Path, vocab: Vocabulary, transform=None):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        caps_dict = read_captions_file(captions_file)
        self.samples: List[Tuple[Path, str]] = []
        for img_name, caps in caps_dict.items():
            for cap in caps:
                img_path = self.images_dir / img_name
                if img_path.is_file():
                    self.samples.append((img_path, cap))

        if not self.samples:
            raise RuntimeError(f"No image+caption samples found. Check {images_dir} and {captions_file}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        token_ids = self.vocab.encode(caption)
        return img, torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch, pad_idx: int):
    """
    Collate function to pad variable-length caption sequences.
    """
    images, seqs = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)

    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s

    return images, padded, torch.tensor(lengths, dtype=torch.long)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int = 256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(base.children())[:-1]  # remove the final fc
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(base.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Optionally freeze CNN
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            # Keep explicit batch dimension even for batch_size=1
            features = self.cnn(x).view(x.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # captions: (batch, seq_len)
        embeddings = self.embed(captions)  # (batch, seq_len, embed_size)
        # prepend image feature as the first "word"
        features = features.unsqueeze(1)  # (batch, 1, embed_size)
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        # Drop the first timestep (image-only) so that
        # outputs align with next-token targets.
        outputs = self.fc(hiddens)[:, 1:, :]
        return outputs

    def generate(self, features, max_len: int, start_idx: int, end_idx: int) -> List[List[int]]:
        """
        Greedy decoding for caption generation.
        """
        batch_size = features.size(0)
        inputs = features.unsqueeze(1)  # (batch, 1, embed_size)
        states = None
        sampled_ids = [[] for _ in range(batch_size)]

        # start tokens
        current_tokens = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=features.device)

        for _ in range(max_len):
            word_embeds = self.embed(current_tokens)  # (batch, 1, embed_size)
            lstm_in = torch.cat((inputs, word_embeds), dim=1)
            hiddens, states = self.lstm(lstm_in, states)
            outputs = self.fc(hiddens[:, -1, :])  # (batch, vocab_size)
            _, predicted = outputs.max(1)
            current_tokens = predicted.unsqueeze(1)

            for i in range(batch_size):
                sampled_ids[i].append(predicted[i].item())

        # stop at <end> when decoding
        cleaned = []
        for seq in sampled_ids:
            if end_idx in seq:
                end_pos = seq.index(end_idx)
                seq = seq[: end_pos + 1]
            cleaned.append(seq)
        return cleaned


class CNNLSTMCaptioner(nn.Module):
    def __init__(self, vocab: Vocabulary, embed_size: int = 256, hidden_size: int = 512):
        super().__init__()
        self.vocab = vocab
        self.encoder = EncoderCNN(embed_size=embed_size)
        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab.word2idx))

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate(self, images, max_len: int = 20) -> List[str]:
        features = self.encoder(images)
        sampled_ids = self.decoder.generate(
            features,
            max_len=max_len,
            start_idx=self.vocab.word2idx["<start>"],
            end_idx=self.vocab.word2idx["<end>"],
        )
        sentences = [self.vocab.decode(seq) for seq in sampled_ids]
        return sentences


# ----------------------------
# 4. Training + evaluation
# ----------------------------

def decode_from_tensor(tensor: torch.Tensor, vocab: Vocabulary) -> str:
    ids = tensor.tolist()
    return vocab.decode(ids)


def train_model(
    images_dir: Path,
    captions_file: Path,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_freq: int = 2,
    val_split: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build vocabulary from captions
    caps_dict = read_captions_file(captions_file)
    all_captions = list(itertools.chain.from_iterable(caps_dict.values()))
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_from_captions(all_captions)
    vocab_path = out_dir / "flickr8k_vocab.json"
    vocab.to_json(vocab_path)
    print(f"[train] Built vocabulary with {len(vocab.word2idx)} tokens, saved to {vocab_path}")

    # 2) Dataset + train/val split
    dataset = Flickr8kCaptionDataset(images_dir, captions_file, vocab)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"[train] Dataset size: train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx),
    )

    # 3) Model, loss, optimizer
    model = CNNLSTMCaptioner(vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_bleu = 0.0
    smooth_fn = SmoothingFunction().method1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_tokens = 0

        for images, captions, lengths in train_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Teacher forcing: input is captions[:, :-1], target is captions[:, 1:]
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)  # (batch, seq_len, vocab)
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            n_tokens += targets.size(0)

        avg_train_loss = total_loss / max(1, n_tokens)

        # ---- Validation + BLEU ----
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        refs = []
        hyps = []

        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)

                # loss
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                outputs = model(images, inputs)
                outputs = outputs.reshape(-1, outputs.size(2))
                targets_flat = targets.reshape(-1)
                loss = criterion(outputs, targets_flat)
                val_loss += loss.item() * targets_flat.size(0)
                val_tokens += targets_flat.size(0)

                # BLEU: generate caption for each image
                generated = model.generate(images, max_len=20)
                for i in range(len(generated)):
                    gt_text = decode_from_tensor(captions[i], vocab)
                    refs.append([gt_text.split()])
                    hyps.append(generated[i].split())

        avg_val_loss = val_loss / max(1, val_tokens)
        bleu4 = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "bleu4": float(bleu4),
        }
        history.append(epoch_stats)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  BLEU-4={bleu4:.4f}"
        )

        # save best model
        if bleu4 > best_bleu:
            best_bleu = bleu4
            model_path = out_dir / "flickr8k_cnn_lstm.pt"
            torch.save(model.state_dict(), model_path)
            print(f"[train] New best BLEU-4={bleu4:.4f}, saved model to {model_path}")

    # Save history and plot
    history_path = out_dir / "flickr8k_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Saved training history to {history_path}")

    plot_training_curves(history, out_dir / "flickr8k_training_curves.png")

    return history


def plot_training_curves(history: List[Dict], out_path: Path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    bleu = [h["bleu4"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Flickr8k CNN+LSTM training/validation loss")

    ax2.plot(epochs, bleu, marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BLEU-4")
    ax2.set_title("Flickr8k BLEU-4 over epochs")

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[plot] Saved training curves (loss + BLEU) to {out_path}")


# ----------------------------
# 5. Main entrypoint (simple CLI interface)
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Flickr8k caption model + generate captions.txt")
    parser.add_argument("--mode", choices=["prepare", "train"], default="train")
    parser.add_argument("--raw_tokens", type=str, help="Path to original Flickr8k.token (for --mode prepare)")
    parser.add_argument("--out_captions", type=str, default="server/data/captions.txt")
    parser.add_argument("--images_dir", type=str, default="server/data/Images")
    parser.add_argument("--captions_file", type=str, default="server/data/captions.txt")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "prepare":
        if not args.raw_tokens:
            raise SystemExit("--raw_tokens must be provided in prepare mode")
        prepare_captions_file(Path(args.raw_tokens), Path(args.out_captions))
    else:
        history = train_model(
            images_dir=Path(args.images_dir),
            captions_file=Path(args.captions_file),
            out_dir=Path(args.out_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            min_freq=args.min_freq,
        )
        print("[done] Training complete. You can now:")
        print(f" - Load {args.out_dir}/flickr8k_cnn_lstm.pt and {args.out_dir}/flickr8k_vocab.json in a new endpoint if desired.")
        print(f" - Use {args.out_dir}/flickr8k_training_curves.png as your accuracy graph.")
        print(f" - Use {args.out_dir}/flickr8k_history.json for detailed metrics and BLEU per epoch.")


if __name__ == "__main__":
    main()
