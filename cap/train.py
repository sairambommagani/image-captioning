"""Zero-shot vs fine-tuned image captioning on Flickr8k (Kaggle / Colab GPU; the tests run a tiny version on CPU).

  python -m cap.train --data /kaggle/input/flickr8k --out artifacts --epochs 3

Steps
  1. captions -> clean -> split by image (6,091 / 1,000 / 1,000), all 5 captions of an image in one split
  2. zero-shot: BLIP-base and GIT-base (both COCO-trained, never trained on Flickr8k)
  3. fine-tune BLIP-base on Flickr8k train (every image-caption pair, mixed precision), best epoch by validation CIDEr
  4. decoding (beam size 1 / 3 / 5) chosen on validation for every model
  5. test (once): BLEU, METEOR, ROUGE-L, CIDEr against the 5 references, CLIPScore, object hallucination, caption
     length and diversity; human agreement (one reference against the other four) as a reference point
  6. deployed = the fine-tuned model if it beats the zero-shot models on validation CIDEr; saved with its processor
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import metrics as M
from . import versioning
from .captioner import FINETUNE_BASE, ZERO_SHOT, Captioner, finetune, open_rgb

BEAMS = (1, 3, 5)


def figure(vdir: Path, keys, img_dir, refs, preds: dict, n=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import textwrap
    except Exception:
        return []
    fig, ax = plt.subplots(n, 2, figsize=(12, 3.2 * n), gridspec_kw={"width_ratios": [1, 1.6]})
    for r, k in enumerate(keys[:n]):
        ax[r, 0].imshow(open_rgb(img_dir / k)); ax[r, 0].axis("off")
        lines = [f"human: {refs[k][0]}"] + [f"{m}: {p[k]}" for m, p in preds.items()]
        ax[r, 1].text(0, 1, "\n".join(textwrap.fill(l, 70) for l in lines), va="top", fontsize=9, family="monospace")
        ax[r, 1].axis("off")
    fig.tight_layout(); fig.savefig(vdir / "examples.png", dpi=80); plt.close(fig)
    return ["examples.png"]


def run(data_dir, art_root, epochs=3, bs=24, lr=2e-5, freeze_vision=False, zero_shot=tuple(ZERO_SHOT), base=FINETUNE_BASE,
        val_subset=500, limit=None, bump="minor", notes="", seed=0, clip_model="openai/clip-vit-base-patch32",
        log=print) -> dict:
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    df, img_dir = D.load(data_dir)
    if limit:
        keep = set(sorted(df["image"].unique())[:limit])
        df = df[df["image"].isin(keep)].reset_index(drop=True)
        sp = D.make_split(df["image"], n_val=max(2, limit // 5), n_test=max(2, limit // 5), seed=seed)
    else:
        sp = D.make_split(df["image"], seed=seed)
    df["split"] = df["image"].map(sp)
    summ = D.summary(df)
    log("data: " + json.dumps(summ))
    assert summ["image_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    refs = D.refs_by_image(df)
    keys = {s: sorted(df.loc[df["split"] == s, "image"].unique()) for s in ("train", "val", "test")}
    vsub = keys["val"][:val_subset]

    def gen(cap, ks, beams):
        return dict(zip(ks, cap.caption([img_dir / k for k in ks], beams=beams)))

    def tune(cap, name):
        """Beam size on validation -> (best beams, CIDEr per beam size)."""
        cur = {}
        for b in BEAMS:
            cur[b] = M.coco_scores(gen(cap, keys["val"], b), refs)[0]["CIDEr"]
            log(f"  {name} beams {b}: val CIDEr {cur[b]:.4f}")
        return max(cur, key=cur.get), cur

    preds, results, decoding = {}, {}, {}
    for name in zero_shot:
        t0 = time.time()
        cap = Captioner(ZERO_SHOT.get(name, name), dev)
        b, curve = tune(cap, name)
        t1 = time.time()
        preds[name] = gen(cap, keys["test"], b)
        sec = time.time() - t1
        decoding[name] = {"beams": b, "val_cider_by_beams": curve}
        results[name] = {"model": ZERO_SHOT.get(name, name), "trained_on_flickr8k": False, "val_cider": curve[b],
                         "gpu_images_per_s" if dev.type == "cuda" else "cpu_images_per_s": round(len(keys["test"]) / sec, 1),
                         "minutes": round((time.time() - t0) / 60, 1)}
        del cap
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    log("fine-tuning BLIP on Flickr8k")
    t0 = time.time()
    tr = keys["train"]
    ft, hist = finetune(base, [img_dir / k for k in tr], [refs[k] for k in tr], dev, epochs=epochs, bs=bs, lr=lr,
                        freeze_vision=freeze_vision, seed=seed, log=log,
                        eval_fn=lambda c: M.coco_scores(gen(c, vsub, 3), refs)[0]["CIDEr"])
    ft_min = round((time.time() - t0) / 60, 1)
    b, curve = tune(ft, "blip_finetuned")
    t1 = time.time()
    preds["blip_finetuned"] = gen(ft, keys["test"], b)
    sec = time.time() - t1
    decoding["blip_finetuned"] = {"beams": b, "val_cider_by_beams": curve}
    results["blip_finetuned"] = {"model": base + " fine-tuned on Flickr8k", "trained_on_flickr8k": True, "val_cider": curve[b],
                                 "train_minutes": ft_min, "epochs": len(hist), "freeze_vision": freeze_vision,
                                 "gpu_images_per_s" if dev.type == "cuda" else "cpu_images_per_s": round(len(keys["test"]) / sec, 1)}

    deployed = "blip_finetuned" if results["blip_finetuned"]["val_cider"] >= max(results[z]["val_cider"] for z in zero_shot) \
        else max(zero_shot, key=lambda z: results[z]["val_cider"])
    log(f"deployed: {deployed} (best validation CIDEr)")
    version, vdir = versioning.new_version(art_root, bump, arch=deployed)
    ft.model.half().save_pretrained(vdir / "model", safe_serialization=True)     # saved before scoring: never lost
    ft.processor.save_pretrained(vdir / "model")
    (vdir / "test_predictions.json").write_text(json.dumps({"references": {k: refs[k] for k in keys["test"]},
                                                            "predictions": preds}, indent=1))
    log(f"  model and test captions saved in {vdir.name}")

    # human agreement: the first reference scored against the other four
    human = {k: refs[k][0] for k in keys["test"]}
    human_refs = {k: refs[k][1:] for k in keys["test"]}
    results["human_reference"] = {"note": "one human caption scored against the other 4 (models are scored against 5)",
                                  "test": M.coco_scores(human, human_refs)[0]}

    log("scoring on the test set")
    imgs = [open_rgb(img_dir / k) for k in keys["test"]]
    per_image = {}
    for name, p in {**preds, "human_reference": human}.items():
        if name != "human_reference":
            sc, per = M.coco_scores(p, refs)
            results[name]["test"] = sc
            per_image[name] = per
            results[name]["hallucination"] = M.hallucination(p, refs)
            results[name]["diversity"] = M.diversity(p)
        else:
            results[name]["diversity"] = M.diversity(p)
            results[name]["hallucination"] = M.hallucination(p, human_refs)
        if clip_model:
            try:
                cs = M.clip_score(imgs, [p[k] for k in keys["test"]], dev, clip_model)
                results[name].setdefault("test", {})["CLIPScore"] = round(float(np.mean(cs)), 4)
            except Exception as e:                         # a metric must never lose a finished training run
                log(f"  CLIPScore skipped for {name}: {e}")
    del imgs
    for name, r in results.items():
        t = r["test"]
        log(f"  {name:16s} BLEU-4 {t.get('BLEU-4')}  METEOR {t.get('METEOR')}  ROUGE-L {t.get('ROUGE-L')}  "
            f"CIDEr {t.get('CIDEr')}  CLIPScore {t.get('CLIPScore')}  hallucination "
            f"{r['hallucination']['captions_with_hallucinated_object_pct']}%  distinct {r['diversity']['distinct_captions_pct']}%")

    worst = sorted(per_image["blip_finetuned"], key=per_image["blip_finetuned"].get)[:15]
    best_ = sorted(per_image["blip_finetuned"], key=per_image["blip_finetuned"].get)[-15:]
    examples = {"worst": [{"image": k, "cider": per_image["blip_finetuned"][k], "caption": preds["blip_finetuned"][k],
                           "zero_shot": preds["blip_zero_shot"].get(k) if "blip_zero_shot" in preds else None,
                           "references": refs[k]} for k in worst],
                "best": [{"image": k, "cider": per_image["blip_finetuned"][k], "caption": preds["blip_finetuned"][k],
                          "references": refs[k]} for k in best_]}
    (vdir / "samples").mkdir()
    rng = np.random.default_rng(seed)
    demo = sorted(rng.choice(keys["test"], min(8, len(keys["test"])), replace=False).tolist())
    for k in demo:
        shutil.copy(img_dir / k, vdir / "samples" / k)
    (vdir / "samples" / "samples.json").write_text(json.dumps(
        [{"file": k, "references": refs[k], **{m: preds[m][k] for m in preds}} for k in demo], indent=1))
    try:
        figs = figure(vdir, demo, img_dir, refs, preds)
    except Exception as e:
        log(f"  figure skipped: {e}")
        figs = []
    metrics = {"deployed": deployed, "base_model": base, "decoding": decoding, "data": summ, "data_sha256": D.fingerprint(df),
               "results": results, "history": hist, "examples": examples, "figures": figs, "epochs": epochs, "batch_size": bs,
               "lr": lr, "minutes": round((time.time() - t_start) / 60, 1)}
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    card = versioning.register(art_root, vdir, version, bump, metrics["data_sha256"], summ, metrics, notes)
    log(f"saved {vdir.name} in {metrics['minutes']} min")
    return {"version": card, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--zero-shot", default=",".join(ZERO_SHOT))
    ap.add_argument("--notes", default="")
    ap.add_argument("--limit", type=int, help="use only N images (quick check)")
    a = ap.parse_args()
    out = run(a.data, a.out, a.epochs, a.batch, a.lr, a.freeze_vision, tuple(z for z in a.zero_shot.split(",") if z),
              limit=a.limit, notes=a.notes, log=lambda *x: print(*x, flush=True))
    print(json.dumps({k: v["test"] for k, v in out["metrics"]["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
