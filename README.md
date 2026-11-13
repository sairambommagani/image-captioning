# Image captioning with a vision-language model (Flickr8k)

Generates a one-sentence description of a photo. Pretrained vision-language models are evaluated zero-shot on
[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (8,091 photos, 5 human captions each), then BLIP is
fine-tuned on it. A FastAPI service captions uploaded photos on a laptop CPU and compares zero-shot with fine-tuned.

**Research prototype.**

> Results (1,000 test photos): fine-tuned BLIP CIDEr **1.062**, BLEU-4 0.361, CLIPScore 0.808; zero-shot GIT CIDEr 0.800,
> zero-shot BLIP 0.652. Fine-tuning: +63% CIDEr over the same model zero-shot. Details in section 4.

## 1. Data

- 8,091 photos (people, dogs, sports, outdoor scenes), 40,455 captions, 5 per photo.
- Split **by photo**: 1,000 test, 1,000 validation, 6,091 train (the standard 6K / 1K / 1K size; this Kaggle copy has
  no official split file, so it is seeded). All 5 captions of a photo stay in one split.
- Captions lower-cased and cleaned (spaces before punctuation, trailing periods).
- Everything (epochs, beam size, deployed model) is chosen on validation; the test set is used once.

## 2. Models

| Model | What it is | Use |
|---|---|---|
| BLIP-base (`Salesforce/blip-image-captioning-base`) | ViT-B/16 image encoder + BERT-style text decoder; pre-trained on 129M image-text pairs, captioning-tuned on COCO | zero-shot, and **fine-tuned on Flickr8k** (deployed) |
| GIT-base (`microsoft/git-base-coco`) | CLIP image encoder + GPT-style decoder, captioning-tuned on COCO | zero-shot comparison |
| Human | one of the 5 captions scored against the other 4 | reference point for "how good can a caption score" |

Fine-tuning: every (photo, caption) pair once per epoch (30K pairs), AdamW lr 2e-5, warm-up + cosine decay, mixed
precision, batch 16, 2 epochs (a 3-epoch run reached validation CIDEr 1.031 vs 1.008 after 2), best epoch by
validation CIDEr. The image tensors are prepared once and kept in memory.

Decoding: beam size 1, 3 or 5, chosen on validation for every model.

## 3. Metrics

| Metric | What it measures |
|---|---|
| BLEU-4 | n-gram precision against the 5 references |
| METEOR | unigram match with stemming and synonyms (NLTK implementation; close to, not identical with, the Java one) |
| ROUGE-L | longest common subsequence with the references |
| **CIDEr** | consensus with the references, tf-idf weighted n-grams (the main captioning metric) |
| CLIPScore | does the caption match the **image** (CLIP ViT-B/32), independent of the references' wording |
| Hallucinated object | share of captions naming a common object (dog, ball, bike, water, ...) that none of the 5 human captions mention |
| Distinct captions | share of different captions over the test set (low = generic, repeated captions) |

## 4. Results

Version `cap_v1.0.0_blip_finetuned_20261113`, trained on a Kaggle T4 GPU (fine-tuning 2 epochs, 63 min; whole run
82 min). Data: 8,091 photos, 40,455 captions, split 6,091 / 1,000 / 1,000 photos, no photo in two splits.

**Validation CIDEr (1,000 photos; decoding chosen here)**

| Model | Beam 1 | Beam 3 | Beam 5 |
|---|---|---|---|
| BLIP zero-shot | **0.608** | 0.592 | 0.596 |
| GIT zero-shot | 0.705 | **0.716** | 0.709 |
| BLIP fine-tuned | 0.997 | 1.032 | **1.037** |

Fine-tuning curve (validation CIDEr on 500 photos, beam 3): epoch 1 **1.032**, epoch 2 **1.053**; training loss 1.93 ->
1.54. An earlier 3-epoch run reached 0.972 / 1.008 / 1.031: the shorter schedule decays the learning rate sooner and
ends slightly higher, so 2 epochs are enough.

**Test (1,000 photos, used once)**

| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | **CIDEr** | CLIPScore | Captions with a hallucinated object | Distinct captions |
|---|---|---|---|---|---|---|---|---|
| BLIP zero-shot (beam 1) | 0.633 | 0.241 | 0.419 | 0.499 | 0.652 | 0.741 | 16.9% | 79.7% |
| GIT zero-shot (beam 3) | 0.696 | 0.263 | 0.469 | 0.514 | 0.800 | 0.787 | 20.6% | 96.0% |
| **BLIP fine-tuned (beam 5, deployed)** | **0.760** | **0.361** | **0.565** | **0.585** | **1.062** | **0.808** | **12.5%** | 96.0% |
| Human (1 caption vs the other 4) | 0.630 | 0.204 | 0.481 | 0.491 | 0.787 | 0.801 | 17.6% | 99.8% |

What the numbers say:

1. **Fine-tuning is the big gain**: CIDEr 0.652 -> 1.062 (+63%) for the same BLIP model, +33% over the best zero-shot
   model (GIT), and better on every metric. Two epochs on 6K photos are enough to adapt a COCO-trained captioner to
   Flickr8k's style and content.
2. **The captions match the photos, not only the references**: CLIPScore (image vs caption, no references) rises from
   0.741 to 0.808, the level of a human caption (0.801). So the reference-metric gain is not just copying the
   annotators' wording.
3. **Fewer hallucinated objects**: 12.5% of fine-tuned captions name a common object no human mentioned, against 16.9%
   (BLIP zero-shot) and 20.6% (GIT). The check is coarse (a fixed object list; humans themselves score 17.6% because
   each annotator mentions details the others skip), but the direction is clear.
4. **Less generic**: zero-shot BLIP repeats itself (only 79.7% distinct captions over 1,000 photos, e.g. the same
   short sentence for many dog photos); the fine-tuned model writes a distinct caption for 96% of photos.
5. **Above the human row on reference metrics** (CIDEr 1.06 vs 0.79): expected and not "better than humans" - the
   model learns the consensus phrasing of the annotators, and a single human caption is scored against 4 references,
   the models against 5. CLIPScore, which ignores wording, puts the model at the human level.
6. **Decoding matters per model**: greedy is best for zero-shot BLIP, beam 3 for GIT, beam 5 for the fine-tuned model;
   choosing it on validation is worth up to +0.04 CIDEr.

## 5. Training on Kaggle, FastAPI on the laptop

Kaggle (`notebooks/run_kaggle.ipynb`, runs unattended with *Save & Run All*): Flickr8k and the code are attached as
inputs; the notebook trains and evaluates everything and writes `results.zip`.

Laptop:
```powershell
pip install -r requirements.txt
Expand-Archive results.zip -DestinationPath artifacts
python -m uvicorn cap.api:app --port 8015            # web page http://localhost:8015, API docs /docs
python -m pytest -q                                  # offline tests (tiny random BLIP, synthetic photos)
```

| Endpoint | Input | Output |
|---|---|---|
| `POST /caption?model=finetuned\|zero_shot\|both` | a photo (JPEG / PNG), optional `beams` | caption(s), beam size, time |
| `GET /samples` | | 8 held-out test photos with their 5 human captions and every model's caption |
| `GET /model-info` | | model card and all results |
| `GET /health` | | status and model version |

The zero-shot model is downloaded from Hugging Face on its first use (about 1 GB). Checks: not an image -> 415,
image under 32 px or unknown model -> 422.

Model versions: `cap_v<MAJOR.MINOR.PATCH>_<model>_<YYYYMMDD>` with a model card (`version.json`: data SHA-256, code
hash, decoding, metrics, file hashes); files are checked before loading.

## 6. Limitations

- Flickr8k is small and narrow (mostly people and dogs outdoors); the fine-tuned model adopts its style and topics.
- Reference-based metrics punish correct captions worded differently from the 5 references; CLIPScore and the
  human row help read them.
- The hallucination check uses a fixed object list with synonyms, not a full object detector.
- One split and one seed.

## Files

| File | What |
|---|---|
| `cap/data.py` | captions, cleaning, split by photo, summary, data fingerprint |
| `cap/captioner.py` | BLIP / GIT loading, generation, fine-tuning |
| `cap/metrics.py` | BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore, hallucination, diversity |
| `cap/train.py` | the full comparison and saving a version |
| `cap/infer.py`, `cap/api.py`, `cap/static/index.html` | inference, FastAPI service, web page |
| `cap/versioning.py` | semantic model versions and model cards |
| `notebooks/run_kaggle.ipynb` | unattended Kaggle run |
| `scripts/make_synthetic.py`, `tests/` | synthetic photos, a tiny random BLIP, tests |
