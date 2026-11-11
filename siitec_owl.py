"""SIITec Hackathon 2569, round 1: OWLv2 defect finder with cached picture features.

Each picture goes through OWLv2's vision part once. After that a new set of
prompts only needs the text part and a small matrix product, so trying prompts
and thresholds takes well under a second.
"""
import base64
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

MODEL = "google/owlv2-base-patch16-ensemble"
RELEASES = "https://github.com/SIITec-Hackathon-2026/round1/releases/download"
IOU_HIT = 0.3        # a found box counts when it overlaps an answer box this much
MAX_PER_CLASS = 10   # boxes kept per class per picture, after merging overlaps
NMS_IOU = 0.3

CLASS_INFO = {  # class -> (part, Thai description)
    "bottle_broken_large": ("bottle", "ปากขวดแตก บิ่นเป็นชิ้นใหญ่"),
    "bottle_broken_small": ("bottle", "ปากขวดบิ่นเล็กน้อย"),
    "bottle_contamination": ("bottle", "มีสิ่งปนเปื้อนในขวด"),
    "metal_nut_bent": ("metal_nut", "ขาของน็อตงอ"),
    "metal_nut_color": ("metal_nut", "มีสีเปื้อนบนน็อต"),
    "metal_nut_flip": ("metal_nut", "น็อตวางคว่ำผิดด้าน (กรอบคลุมทั้งชิ้น)"),
    "metal_nut_scratch": ("metal_nut", "รอยขีดข่วนบนน็อต"),
    "screw_defect": ("screw", "สกรูมีตำหนิ ที่หัว คอ เกลียว หรือปลาย"),
}
PART_TH = {"bottle": "ขวด", "metal_nut": "น็อตโลหะ", "screw": "สกรู"}


# ---------------------------------------------------------------- data

def _release_url(name):
    return f"{RELEASES}/{'data' if name == 'train' else 'test'}/{name}.zip"


def get_data(name, folder="."):
    """Return the folder siitec_round1_<name>, unzipping <name>.zip (from SIITEC_DATA, here, or Releases)."""
    top = Path(folder) / f"siitec_round1_{name}"
    if top.exists():
        return top
    local = Path(os.environ.get("SIITEC_DATA", folder)) / f"{name}.zip"
    if not local.exists():
        local = Path(folder) / f"{name}.zip"
    if not local.exists():
        urllib.request.urlretrieve(_release_url(name), local)
    with zipfile.ZipFile(local) as z:
        z.extractall(folder)
    return top


def test_available(folder="."):
    """The test set (with answers) is published only when the round-1 results are out."""
    if (Path(folder) / "siitec_round1_test").exists():
        return True
    if any((Path(d) / "test.zip").exists() for d in (os.environ.get("SIITEC_DATA", folder), folder)):
        return True
    try:
        req = urllib.request.Request(_release_url("test"), method="HEAD")
        return urllib.request.urlopen(req, timeout=5).status == 200
    except Exception:
        return False


def load_set(name, folder="."):
    """-> (image folder, records, classes); records: file, width, height, category, boxes."""
    top = get_data(name, folder)
    data = json.loads((top / "labels.json").read_text())
    return top / "images", data["images"], data["classes"]


def read_rgb(path):
    """A picture as a numpy RGB array (what my_filter and my_draw receive)."""
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))


# ---------------------------------------------------------------- model

class Finder:
    def __init__(self, device=None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()      # no weight-loading bars or warnings in the notebook
        hf_logging.disable_progress_bar()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(self.device, self.dtype).eval()
        self._text = {}

    @torch.no_grad()
    def embed_pictures(self, folder, records, batch=8, progress=None):
        """Vision part once per picture: per-patch class embeddings, logit shift/scale, boxes in pixels."""
        m, out = self.model, {"emb": [], "shift": [], "scale": [], "boxes": []}
        for i in range(0, len(records), batch):
            part = records[i:i + batch]
            imgs = [Image.open(Path(folder) / r["file"]).convert("RGB") for r in part]
            px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            fmap = m.image_embedder(pixel_values=px)[0]
            feats = fmap.reshape(fmap.shape[0], -1, fmap.shape[-1])
            head = m.class_head
            emb = head.dense0(feats)
            out["emb"].append(emb / (emb.norm(dim=-1, keepdim=True) + 1e-6))
            out["shift"].append(head.logit_shift(feats)[..., 0])
            out["scale"].append(head.elu(head.logit_scale(feats))[..., 0] + 1)
            cx, cy, w, h = m.box_predictor(feats, fmap).float().unbind(-1)
            side = torch.tensor([max(r["width"], r["height"]) for r in part], device=self.device)[:, None]
            out["boxes"].append(torch.stack([(cx - w / 2) * side, (cy - h / 2) * side,
                                             (cx + w / 2) * side, (cy + h / 2) * side], -1))
            if progress:
                progress(min(i + batch, len(records)), len(records))
        return {k: torch.cat(v) for k, v in out.items()}

    @torch.no_grad()
    def text(self, prompts):
        new = [p for p in dict.fromkeys(prompts) if p not in self._text]
        if new:
            # OWLv2 reads at most 16 tokens of a prompt; longer ones are cut there
            t = self.proc(text=new, return_tensors="pt", padding="max_length", truncation=True,
                          max_length=16).to(self.device)
            f = self.model.owlv2.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
            f = getattr(f, "pooler_output", f)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
            self._text.update(zip(new, f.to(self.dtype)))
        return torch.stack([self._text[p] for p in prompts])

    @torch.no_grad()
    def find(self, cache, records, prompts):
        """prompts: {class: [prompt, ...]}; a prompt starting with '-' is a 'not this' prompt.
        Returns, per picture, every candidate box: [{"label", "score", "box"}] (no threshold yet)."""
        out = [[] for _ in records]
        parts = sorted({CLASS_INFO[c][0] for c in prompts})
        for part in parts:
            idx = [i for i, r in enumerate(records) if r["category"] == part]
            classes = [c for c in prompts if CLASS_INFO[c][0] == part and any(p.strip() for p in prompts[c])]
            if not idx or not classes:
                continue
            ii = torch.tensor(idx, device=self.device)
            emb, shift, scale, boxes = (cache[k][ii] for k in ("emb", "shift", "scale", "boxes"))

            def scores(texts):
                q = self.text(texts)                                   # Q x D
                logit = (torch.einsum("npd,qd->npq", emb, q) + shift[..., None]) * scale[..., None]
                return logit.float().sigmoid()                         # N x P x Q

            negs = [p[1:].strip() for c in classes for p in prompts[c] if p.strip().startswith("-") and p[1:].strip()]
            neg = scores(negs).amax(-1) if negs else None
            for c in classes:
                pos = [p.strip() for p in prompts[c] if p.strip() and not p.strip().startswith("-")]
                if not pos:
                    continue
                s = scores(pos).amax(-1)                               # N x P, best prompt per patch
                if neg is not None:
                    s = torch.where(neg > s, torch.zeros_like(s), s)
                # best 100 patches per picture, then overlap merging for all pictures in one call
                v, top = s.topk(min(100, s.shape[1]), dim=1)
                b = torch.gather(boxes, 1, top[..., None].expand(-1, -1, 4))
                pic = torch.arange(len(idx), device=s.device)[:, None].expand_as(v)
                v, b, pic = v.reshape(-1), b.reshape(-1, 4), pic.reshape(-1)
                keep = torchvision.ops.batched_nms(b, v, pic, NMS_IOU)
                rows = zip(pic[keep].tolist(), v[keep].tolist(), b[keep].tolist())
                count = [0] * len(idx)
                for n, sc, bx in rows:                 # keep is sorted by score
                    if count[n] < MAX_PER_CLASS and sc > 0.005:
                        count[n] += 1
                        out[idx[n]].append({"label": c, "score": round(sc, 4),
                                            "box": [round(x, 1) for x in bx]})
        return out


def apply_thresholds(candidates, thresholds):
    return [[d for d in ds if d["score"] >= thresholds.get(d["label"], 1.0)] for ds in candidates]


def apply_filter(fn, folder, records, detections):
    """Run the team's my_filter(image, detections) on every picture; image is numpy RGB,
    detections are all candidates (the team's code picks its own score thresholds)."""
    if fn is None:
        return detections
    out = []
    for r, ds in zip(records, detections):
        res = fn(read_rgb(Path(folder) / r["file"]), [dict(d, box=list(d["box"])) for d in ds])
        out.append(check_detections(res, r["file"]))
    return out


def check_detections(res, picture):
    """my_filter's answer for one picture, checked: a list of {"label", "score", "box"}.
    Raises FilterOutputError with a message the team can act on."""
    where = f"(ภาพ {picture})"
    if res is None:
        raise FilterOutputError(f"my_filter ไม่ได้ return อะไรเลย — ต้อง return list ของกรอบ {where}")
    if isinstance(res, (dict, str)) or not hasattr(res, "__iter__"):
        raise FilterOutputError(f"my_filter ต้อง return list ของกรอบ แต่ได้ {type(res).__name__} {where}")
    out = []
    for d in res:
        if not isinstance(d, dict):
            raise FilterOutputError(f"กรอบแต่ละอันต้องเป็น dict แต่ได้ {type(d).__name__} {where}")
        missing = [k for k in ("label", "score", "box") if k not in d]
        if missing:
            raise FilterOutputError(f"กรอบขาด {', '.join(missing)} {where}")
        if d["label"] not in CLASS_INFO:
            raise FilterOutputError(f"ไม่มีแบบชื่อ {d['label']!r} {where}")
        try:
            score = float(d["score"])
            box = [float(v) for v in d["box"]]
        except (TypeError, ValueError):
            raise FilterOutputError(f"score ต้องเป็นตัวเลข และ box ต้องเป็นตัวเลข 4 ตัว {where}") from None
        if len(box) != 4 or not all(np.isfinite(box + [score])):
            raise FilterOutputError(f"box ต้องเป็น [x1, y1, x2, y2] ตัวเลข 4 ตัว {where}")
        x1, y1, x2, y2 = box
        out.append({"label": d["label"], "score": score,
                    "box": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
    return out


class FilterOutputError(ValueError):
    pass


def load_code(source, name):
    """Run exported cell source in a fresh namespace and return the function `name` (or None)."""
    if not source.strip():
        return None
    ns = {"np": np, "cv2": cv2}
    exec(compile(source, "<script>", "exec"), ns)
    return ns.get(name)


# ---------------------------------------------------------------- scoring

def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(found, answers):
    """Greedy by score, per class. Returns (hit flags for found, hit flags for answers)."""
    f_hit, a_hit = [False] * len(found), [False] * len(answers)
    for i in sorted(range(len(found)), key=lambda i: -found[i]["score"]):
        best, bj = IOU_HIT, -1
        for j, a in enumerate(answers):
            if not a_hit[j] and a["label"] == found[i]["label"]:
                v = _iou(found[i]["box"], a["box"])
                if v >= best:
                    best, bj = v, j
        if bj >= 0:
            f_hit[i] = a_hit[bj] = True
    return f_hit, a_hit


def evaluate(records, detections, classes):
    """Score = mean F1 over classes. Returns (score, {class: {tp, fp, fn, p, r, f1}}, per-picture marks)."""
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    marks = []
    for r, ds in zip(records, detections):
        f_hit, a_hit = match(ds, r["boxes"])
        for d, h in zip(ds, f_hit):
            if d["label"] in per:
                per[d["label"]]["tp" if h else "fp"] += 1
        for a, h in zip(r["boxes"], a_hit):
            if not h:
                per[a["label"]]["fn"] += 1
        marks.append((f_hit, a_hit))
    for v in per.values():
        v["p"] = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        v["r"] = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        v["f1"] = 2 * v["tp"] / (2 * v["tp"] + v["fp"] + v["fn"]) if v["tp"] + v["fp"] + v["fn"] else 1.0
    return sum(v["f1"] for v in per.values()) / len(per), per, marks


# ---------------------------------------------------------------- submission

def make_export(team, prompts, script, train_score=None):
    """What a team hands in (with its video): prompts and script, enough to re-run on any set."""
    return {
        "kind": "siitec-round1", "version": 2, "team": team,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL,
        "prompts": prompts, "script": script, "train_score": train_score,
    }


def thumbnail_b64(path, size=360):
    img = Image.fromarray(path) if isinstance(path, np.ndarray) else Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()
