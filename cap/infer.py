"""Inference: image -> caption (CPU), with the fine-tuned model and, on request, the zero-shot model for comparison."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .captioner import ZERO_SHOT, Captioner
from .versioning import resolve, verify

MAX_BYTES = 20_000_000
DISCLAIMER = ("Research prototype. Fine-tuned on Flickr8k (everyday photos of people and animals); captions for other "
              "kinds of images (documents, medical, screenshots) are unreliable.")


class Service:
    def __init__(self, root: Path):
        self.dir = resolve(Path(root))
        bad = verify(self.dir)
        if bad:
            raise RuntimeError(f"model files changed since training: {bad}")
        self.card = json.loads((self.dir / "version.json").read_text())
        self.metrics = json.loads((self.dir / "metrics.json").read_text())
        self.models = {"finetuned": Captioner(self.dir / "model")}
        self.beams = {"finetuned": self.metrics["decoding"]["blip_finetuned"]["beams"],
                      "zero_shot": self.metrics["decoding"].get("blip_zero_shot", {}).get("beams", 3)}
        f = self.dir / "samples" / "samples.json"
        self.samples = json.loads(f.read_text()) if f.exists() else []

    def model(self, name):
        if name not in self.models:                       # zero-shot BLIP: downloaded from Hugging Face on first use
            self.models[name] = Captioner(ZERO_SHOT["blip_zero_shot"])
        return self.models[name]

    def caption(self, data: bytes, which: str = "finetuned", beams: int | None = None) -> dict:
        t0 = time.time()
        if not data:
            raise ValueError("empty file")
        if len(data) > MAX_BYTES:
            raise ValueError("file larger than 20 MB")
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except (UnidentifiedImageError, OSError):
            raise ValueError("not an image (upload a JPEG or PNG photo)")
        im = ImageOps.exif_transpose(im).convert("RGB")
        if min(im.size) < 32:
            raise ValueError(f"image too small ({im.size[0]}x{im.size[1]})")
        names = ["finetuned", "zero_shot"] if which == "both" else [which]
        if any(n not in ("finetuned", "zero_shot") for n in names):
            raise ValueError("model must be finetuned, zero_shot or both")
        out = {"results": {}}
        for n in names:
            t1 = time.time()
            b = beams or self.beams[n]
            out["results"][n] = {"caption": self.model(n).caption([im], beams=b)[0], "beams": b,
                                 "seconds": round(time.time() - t1, 3)}
        ft = self.metrics["results"]["blip_finetuned"]["test"]
        out.update({"model_version": self.card["name"], "image_size": list(im.size),
                    "reliability": (f"fine-tuned model on 1,000 Flickr8k test images: CIDEr {ft['CIDEr']}, BLEU-4 "
                                    f"{ft['BLEU-4']}" + (f", CLIPScore {ft['CLIPScore']}" if ft.get("CLIPScore") else "")),
                    "disclaimer": DISCLAIMER, "seconds": round(time.time() - t0, 3)})
        return out
