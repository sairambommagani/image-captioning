"""Tiny fake Flickr8k (Images/ + captions.txt in the Kaggle layout) and a tiny random BLIP model, for offline tests.

  python -m scripts.make_synthetic out_dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCENES = [("red", (200, 40, 40), "a dog runs on the grass"), ("blue", (40, 60, 200), "a man surfs in the water"),
          ("white", (235, 235, 235), "a child plays in the snow"), ("yellow", (220, 200, 60), "a girl sits on the sand")]


def write(root: Path, n: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "Images").mkdir(parents=True, exist_ok=True)
    lines = ["image,caption"]
    for i in range(n):
        name, col, cap = SCENES[i % len(SCENES)]
        im = Image.new("RGB", (160, 120), col)
        d = ImageDraw.Draw(im)
        x, y = rng.integers(10, 100, 2)
        d.ellipse([x, y, x + 30, y + 20], fill=(30, 30, 30))
        f = f"{1000 + i}_{name}.jpg"
        im.save(root / "Images" / f)
        for j in range(5):
            extra = ["", " , quickly", " .", " outside", " today"][j]
            lines.append(f"{f},{cap.capitalize()}{extra}")
    (root / "captions.txt").write_text("\n".join(lines) + "\n")
    return root


def tiny_blip(out: Path, texts) -> Path:
    """A 1-layer BLIP with a word-level BERT tokenizer built from the captions (no download)."""
    from transformers import (BertTokenizer, BlipConfig, BlipForConditionalGeneration, BlipImageProcessor,
                              BlipProcessor)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    words = sorted({w for t in texts for w in t.lower().replace(",", " ").replace(".", " ").split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[DEC]"] + words
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = BertTokenizer(str(out / "vocab.txt"), do_lower_case=True, bos_token="[DEC]")
    proc = BlipProcessor(image_processor=BlipImageProcessor(size={"height": 64, "width": 64}), tokenizer=tok)
    cfg = BlipConfig(vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                        image_size=64, patch_size=16),
                     text_config=dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                      num_attention_heads=2, encoder_hidden_size=32, bos_token_id=5, pad_token_id=0,
                                      sep_token_id=3, eos_token_id=3, max_position_embeddings=64))
    m = BlipForConditionalGeneration(cfg)
    m.config.text_config.bos_token_id = 5
    m.save_pretrained(out)
    proc.save_pretrained(out)
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic")))
