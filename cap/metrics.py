"""Caption metrics.

Reference-based (all 5 human captions are references): BLEU-1..4, METEOR, ROUGE-L, CIDEr (consensus-based, the main
captioning metric; tf-idf weighted n-gram similarity to the references, x10 scale as usual).
Implementations: pycocoevalcap for BLEU, ROUGE-L and CIDEr (pure Python), NLTK for METEOR (Java is not needed;
values are close to, not identical with, the official Java METEOR). Tokenisation: lower case, punctuation removed.

Reference-free: CLIPScore (Hessel et al. 2021) = 2.5 x max(cos(image, caption), 0) with CLIP ViT-B/32: does the
caption match the image, independent of the references' wording.

Diagnostics:
  - object hallucination (CHAIR-style): share of captions naming an object from a list of common Flickr8k objects
    that none of the 5 references mention (with synonyms)
  - length, vocabulary used, share of distinct captions (generic captions repeat across images)
"""
from __future__ import annotations

import re

import numpy as np


def tok(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def coco_scores(preds: dict[str, str], refs: dict[str, list[str]]) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    keys = sorted(preds)
    gts = {k: [tok(r) for r in refs[k]] for k in keys}
    res = {k: [tok(preds[k]) or "empty"] for k in keys}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    r, _ = Rouge().compute_score(gts, res)
    c, per = Cider().compute_score(gts, res)
    out = {"BLEU-1": b[0], "BLEU-2": b[1], "BLEU-3": b[2], "BLEU-4": b[3], "ROUGE-L": r, "CIDEr": c}
    out["METEOR"] = meteor(preds, refs)
    return {k: (round(float(v), 4) if v is not None else None) for k, v in out.items()}, dict(zip(keys, np.round(per, 4).tolist()))


def meteor(preds, refs):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        vals = [meteor_score([tok(r).split() or ["empty"] for r in refs[k]], tok(p).split() or ["empty"]) for k, p in preds.items()]
        return float(np.mean(vals))
    except Exception:
        return None


# object words and synonyms (Flickr8k is mostly people, dogs, sports and outdoor scenes)
OBJECTS = {
    "dog": ["dog", "dogs", "puppy", "puppies"], "cat": ["cat", "cats", "kitten"], "horse": ["horse", "horses"],
    "bird": ["bird", "birds"], "cow": ["cow", "cows"], "sheep": ["sheep"],
    "man": ["man", "men", "guy", "guys", "gentleman"], "woman": ["woman", "women", "lady", "ladies"],
    "boy": ["boy", "boys"], "girl": ["girl", "girls"], "child": ["child", "children", "kid", "kids", "toddler", "baby"],
    "ball": ["ball", "balls", "football", "soccer ball", "basketball", "tennis ball"],
    "bike": ["bike", "bikes", "bicycle", "bicycles", "cyclist"], "motorcycle": ["motorcycle", "motorbike", "dirt bike"],
    "car": ["car", "cars"], "boat": ["boat", "boats", "kayak", "canoe"], "skateboard": ["skateboard", "skateboarder"],
    "surfboard": ["surfboard", "surfer"], "snowboard": ["snowboard", "snowboarder"], "skis": ["ski", "skis", "skier"],
    "frisbee": ["frisbee"], "stick": ["stick"], "rope": ["rope"], "hat": ["hat", "cap", "helmet"],
    "umbrella": ["umbrella"], "bench": ["bench"], "tree": ["tree", "trees"], "water": ["water", "lake", "river", "pool", "ocean", "sea"],
    "snow": ["snow", "snowy"], "sand": ["sand", "beach"], "grass": ["grass", "grassy", "field"], "rock": ["rock", "rocks", "cliff"],
    "street": ["street", "road", "sidewalk"], "building": ["building", "buildings"], "fence": ["fence"],
    "camera": ["camera"], "guitar": ["guitar"], "toy": ["toy", "toys"],
}
_LOOKUP = sorted(((w, k) for k, ws in OBJECTS.items() for w in ws), key=lambda x: -len(x[0]))


def objects_in(text: str) -> set[str]:
    t = " " + tok(text) + " "
    found = set()
    for w, k in _LOOKUP:
        if f" {w} " in t:
            found.add(k)
    return found


def hallucination(preds: dict[str, str], refs: dict[str, list[str]]) -> dict:
    n_obj = n_bad = n_cap_bad = 0
    examples = []
    for k, p in preds.items():
        mentioned = objects_in(p)
        allowed = set().union(*[objects_in(r) for r in refs[k]])
        bad = mentioned - allowed
        n_obj += len(mentioned); n_bad += len(bad); n_cap_bad += bool(bad)
        if bad and len(examples) < 20:
            examples.append({"image": k, "caption": p, "not_in_references": sorted(bad)})
    return {"captions_with_hallucinated_object_pct": round(100 * n_cap_bad / max(len(preds), 1), 2),
            "hallucinated_object_mentions_pct": round(100 * n_bad / max(n_obj, 1), 2), "examples": examples}


def diversity(preds: dict[str, str]) -> dict:
    caps = [tok(p) for p in preds.values()]
    words = [w for c in caps for w in c.split()]
    return {"mean_words": round(float(np.mean([len(c.split()) for c in caps])), 2),
            "distinct_captions_pct": round(100 * len(set(caps)) / max(len(caps), 1), 2),
            "vocabulary_used": len(set(words))}


def clip_score(images: list, captions: list[str], dev, model_name="openai/clip-vit-base-patch32", bs=64) -> list[float]:
    """images: list of PIL images. Returns CLIPScore per pair (2.5 x max(cos, 0))."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(model_name).to(dev).eval()
    p = CLIPProcessor.from_pretrained(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = p(text=["A photo depicts " + c for c in captions[i:i + bs]], images=images[i:i + bs], return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(dev)
            o = m(**x)                                   # image_embeds / text_embeds: projected (stable across versions)
            ie = torch.nn.functional.normalize(o.image_embeds, dim=-1)
            te = torch.nn.functional.normalize(o.text_embeds, dim=-1)
            out += (2.5 * (ie * te).sum(-1).clamp(min=0)).cpu().tolist()
    return out
