"""Vision-language captioning models (Hugging Face), one interface for zero-shot use, fine-tuning and serving.

  Salesforce/blip-image-captioning-base   BLIP (ViT-B/16 image encoder + BERT-style text decoder), captioning model
                                          pre-trained on 129M image-text pairs and fine-tuned on COCO captions
  microsoft/git-base-coco                 GIT (CLIP image encoder + GPT-style decoder), fine-tuned on COCO captions
Both are used zero-shot on Flickr8k (never trained on it); BLIP is then fine-tuned on Flickr8k.
"""
from __future__ import annotations

import numpy as np
import torch

ZERO_SHOT = {"blip_zero_shot": "Salesforce/blip-image-captioning-base", "git_zero_shot": "microsoft/git-base-coco"}
FINETUNE_BASE = "Salesforce/blip-image-captioning-base"


def open_rgb(x):
    from PIL import Image, ImageOps
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    im = Image.open(x)
    return ImageOps.exif_transpose(im).convert("RGB")


class Captioner:
    def __init__(self, name_or_path: str, dev=None, dtype=None):
        from transformers import AutoProcessor
        self.dev = dev or torch.device("cpu")
        self.path = str(name_or_path)
        self.processor = AutoProcessor.from_pretrained(name_or_path)
        if "git" in self.path.lower() and "blip" not in self.path.lower():
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float32)
            self.kind = "git"
        else:
            from transformers import BlipForConditionalGeneration
            self.model = BlipForConditionalGeneration.from_pretrained(name_or_path, torch_dtype=torch.float32)
            self.kind = "blip"
        self.model.to(self.dev).eval()

    def pixels(self, images):
        """PIL images or file paths -> pixel tensor."""
        return self.processor(images=[open_rgb(x) for x in images], return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def caption(self, images, beams=3, max_new_tokens=30, bs=16, length_penalty=1.0) -> list[str]:
        out = []
        for i in range(0, len(images), bs):
            pv = self.pixels(images[i:i + bs]).to(self.dev)
            with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                g = self.model.generate(pixel_values=pv, num_beams=beams, max_new_tokens=max_new_tokens,
                                        length_penalty=length_penalty, early_stopping=beams > 1)
            out += [t.strip() for t in self.processor.batch_decode(g, skip_special_tokens=True)]
        return out


def finetune(base: str, images: list, captions_per_image: list[list[str]], dev, epochs=3, bs=24, lr=2e-5,
             freeze_vision=False, eval_fn=None, seed=0, log=print):
    """images: train image paths; captions_per_image: their 5 captions. Every (image, caption) pair once per epoch.
    The image tensors are computed once (fp16, in memory), so an epoch is pure GPU work."""
    import time
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cap = Captioner(base, dev)
    model, proc = cap.model, cap.processor
    t0 = time.time()
    # images cached once as uint8 at the model's input size (6,091 x 384 x 384 x 3 = 2.7 GB, fits a 12 GB Colab);
    # resize as the processor does, rescale and normalise on the GPU per batch
    ip = proc.image_processor
    sz = ip.size
    size = int(sz["height"] if hasattr(sz, "__getitem__") and not isinstance(sz, int) else getattr(sz, "height", sz))
    mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
    from PIL import Image
    pix = np.empty((len(images), size, size, 3), np.uint8)
    for i, p in enumerate(images):
        pix[i] = np.asarray(open_rgb(p).resize((size, size), Image.BICUBIC))
    log(f"  cached {len(pix)} images {pix.shape[1:]} uint8 ({pix.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f} s")
    pairs = [(i, c) for i, cs in enumerate(captions_per_image) for c in cs]
    if freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(pairs) / bs))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s / steps, 1)))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        order = rng.permutation(len(pairs))
        for k in range(0, len(order), bs):
            b = [pairs[j] for j in order[k:k + bs]]
            tok = proc.tokenizer([c for _, c in b], padding=True, truncation=True, max_length=40, return_tensors="pt")
            ids = tok["input_ids"].to(dev)
            if cap.kind == "blip":           # BLIP's decoder starts from [DEC] (as in generation), not the tokenizer's [CLS]
                ids[:, 0] = model.config.text_config.bos_token_id
            labels = ids.masked_fill(tok["attention_mask"].to(dev) == 0, -100)
            pv = torch.from_numpy(pix[[i for i, _ in b]]).to(dev).permute(0, 3, 1, 2).float().div_(255).sub_(mean).div_(std)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = model(pixel_values=pv, input_ids=ids, attention_mask=tok["attention_mask"].to(dev), labels=labels).loss
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                sched.step(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if n % 200 == 0:
                log(f"    step {n}: loss {tot / n:.4f} ({time.time() - t0:.0f} s)")
        rec = {"epoch": ep + 1, "train_loss": round(tot / max(n, 1), 4), "seconds": round(time.time() - t0, 1)}
        if eval_fn:
            model.eval()
            rec["val_cider"] = eval_fn(cap)
            if rec["val_cider"] > best:
                best = rec["val_cider"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(rec)
        log(f"  fine-tune epoch {ep + 1}/{epochs}: loss {rec['train_loss']:.4f}" +
            (f"  val CIDEr {rec['val_cider']:.4f}" if "val_cider" in rec else "") + f"  ({rec['seconds']:.0f} s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return cap, hist
