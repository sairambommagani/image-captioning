"""Flickr8k (Kaggle: adityajn105/flickr8k): images and captions, split, caption cleaning.

  Images/<image>.jpg   8,091 photos (people and animals in everyday scenes)
  captions.txt         image,caption - 5 human captions per image (40,455 lines)

Split: 6,000 / 1,000 / 1,000 images is the standard size (Karpathy split); this Kaggle copy has no split file and
8,091 images, so the split is made here by image, seeded: 1,000 test, 1,000 validation, the remaining 6,091 train.
All 5 captions of an image stay in the same split (no image is seen in training and tested).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_root(data_dir: Path) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    caps = sorted(data_dir.rglob("captions.txt"), key=lambda p: len(p.parts))
    if not caps:
        raise FileNotFoundError(f"captions.txt not found under {data_dir}")
    img = caps[0].parent / "Images"
    if not img.is_dir():
        cand = [p for p in data_dir.rglob("*") if p.is_dir() and p.name.lower() == "images"]
        if not cand:
            raise FileNotFoundError(f"Images folder not found under {data_dir}")
        img = cand[0]
    return caps[0], img


def clean(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+([.,!?;:])", r"\1", c)
    c = re.sub(r"\s+", " ", c).strip(" .")
    return c


def load(data_dir: Path) -> tuple[pd.DataFrame, Path]:
    cap_file, img_dir = find_root(data_dir)
    rows = []
    with open(cap_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 and line.lower().startswith("image,"):
                continue
            if "," not in line:
                continue
            img, cap = line.split(",", 1)              # captions themselves contain commas
            img = img.split("#")[0].strip()
            if cap.strip():
                rows.append((img, clean(cap.strip().strip('"'))))
    df = pd.DataFrame(rows, columns=["image", "caption"])
    df = df[df["image"].map(lambda x: (img_dir / x).exists())].reset_index(drop=True)
    return df, img_dir


def make_split(images, n_val=1000, n_test=1000, seed=0) -> dict:
    ims = np.array(sorted(set(images)))
    if len(ims) < 2 * (n_val + n_test):          # small data (tests): 20% validation, 20% test
        n_val = n_test = max(1, len(ims) // 5)
    perm = np.random.default_rng(seed).permutation(len(ims))
    out = {}
    for k, i in enumerate(perm):
        out[ims[i]] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return out


def refs_by_image(df: pd.DataFrame) -> dict[str, list[str]]:
    return {k: list(g) for k, g in df.groupby("image")["caption"]}


def summary(df: pd.DataFrame) -> dict:
    words = df["caption"].str.split().str.len()
    out = {"images": int(df["image"].nunique()), "captions": int(len(df)),
           "captions_per_image": df.groupby("image").size().value_counts().to_dict(),
           "caption_words": {"mean": round(float(words.mean()), 2), "median": float(words.median()), "max": int(words.max())},
           "vocabulary": int(len(set(" ".join(df["caption"]).split())))}
    if "split" in df:
        out["by_split"] = {k: {"images": int(g["image"].nunique()), "captions": int(len(g))} for k, g in df.groupby("split")}
        s = {k: set(g["image"]) for k, g in df.groupby("split")}
        out["image_overlap"] = {"train_val": len(s.get("train", set()) & s.get("val", set())),
                                "train_test": len(s.get("train", set()) & s.get("test", set())),
                                "val_test": len(s.get("val", set()) & s.get("test", set()))}
    out["captions_per_image"] = {int(k): int(v) for k, v in out["captions_per_image"].items()}
    return out


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for r in df.sort_values(["image", "caption"]).itertuples(index=False):
        h.update(f"{r.image}|{r.caption}|{getattr(r, 'split', '')}\n".encode())
    return h.hexdigest()
