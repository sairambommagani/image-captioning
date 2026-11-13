"""Tests on a tiny synthetic Flickr8k with a tiny random BLIP (no download, no GPU)."""
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cap import data as D                                    # noqa: E402
from cap import metrics as M                                 # noqa: E402


def test_load_clean_split(tmp_path):
    from scripts.make_synthetic import write
    df, img_dir = D.load(write(tmp_path))
    assert len(df) == 200 and df["image"].nunique() == 40
    assert df["caption"].iloc[1] == "a dog runs on the grass, quickly"            # lower case, spaces before , fixed
    df["split"] = df["image"].map(D.make_split(df["image"]))
    s = D.summary(df)
    assert s["image_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0} and set(s["by_split"]) == {"train", "val", "test"}


def test_metrics():
    refs = {"a": ["a dog runs on the grass", "a brown dog running", "dog on grass", "a dog plays", "the dog runs"],
            "b": ["a man surfs a wave", "a surfer in the ocean", "man on a surfboard", "a person surfing", "surfing man"]}
    good, sc_bad = M.coco_scores({"a": "a dog runs on the grass", "b": "a man surfs a wave"}, refs)[0], \
        M.coco_scores({"a": "a man surfs a wave", "b": "a dog runs on the grass"}, refs)[0]
    assert good["CIDEr"] > sc_bad["CIDEr"] and good["BLEU-4"] > sc_bad["BLEU-4"]
    h = M.hallucination({"a": "a dog with a frisbee", "b": "a man surfs"}, refs)
    assert h["captions_with_hallucinated_object_pct"] == 50.0 and h["examples"][0]["not_in_references"] == ["frisbee"]
    assert M.diversity({"a": "x y", "b": "x y"})["distinct_captions_pct"] == 50.0


def test_pipeline_tiny_and_api(tmp_path, monkeypatch):
    from scripts.make_synthetic import tiny_blip, write
    from cap import train, versioning
    data = write(tmp_path / "data")
    df, _ = D.load(data)
    tiny = str(tiny_blip(tmp_path / "tiny", df["caption"].tolist()))
    out = train.run(data, tmp_path / "artifacts", epochs=2, bs=8, lr=1e-3, zero_shot=(tiny,), base=tiny, val_subset=8,
                    clip_model=None, notes="test", log=lambda *a: None)
    card = out["version"]
    vdir = tmp_path / "artifacts" / card["name"]
    for f in ("model/config.json", "metrics.json", "test_predictions.json", "samples/samples.json", "version.json"):
        assert (vdir / f).exists(), f
    assert versioning.verify(vdir) == []
    m = json.loads((vdir / "metrics.json").read_text())
    assert {"blip_finetuned", "human_reference"} <= set(m["results"]) and m["decoding"]["blip_finetuned"]["beams"] in (1, 3, 5)

    monkeypatch.setenv("CAP_ARTIFACTS", str(tmp_path / "artifacts"))
    import importlib
    import cap.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    jpg = next((vdir / "samples").glob("*.jpg")).read_bytes()
    with TestClient(api.app) as c:
        r = c.post("/caption", files={"file": ("x.jpg", jpg, "image/jpeg")})
        assert r.status_code == 200, r.text
        j = r.json()
        assert isinstance(j["results"]["finetuned"]["caption"], str) and j["model_version"] == card["name"]
        assert c.post("/caption", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
        assert c.post("/caption?model=nope", files={"file": ("x.jpg", jpg, "image/jpeg")}).status_code == 422
        buf = io.BytesIO(); Image.new("RGB", (10, 10)).save(buf, format="PNG")
        assert c.post("/caption", files={"file": ("s.png", buf.getvalue(), "image/png")}).status_code == 422
        s = c.get("/samples").json()
        assert len(s) == 8 and len(s[0]["references"]) == 5
        assert c.get("/samples/" + s[0]["file"]).status_code == 200
        assert c.get("/samples/../metrics.json").status_code == 404
        assert c.get("/model-info").json()["version"]["name"] == card["name"]
        assert "Image captioning" in c.get("/").text
