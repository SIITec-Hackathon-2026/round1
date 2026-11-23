"""SIITec Hackathon 2569 - OWLv2 playground: type what to look for, see where the AI finds it.

A warm-up before the round-1 task: everyday pictures (people, cars, dogs ...) or the team's own,
one prompt per line, a confidence slider, and the boxes drawn on the picture.
Play().show() builds the screen in a notebook (Colab or local Jupyter).
"""
import base64
import html
import io
import urllib.request
from pathlib import Path

import cv2
import ipywidgets as W
import numpy as np
import torch
import torchvision
from IPython.display import display
from PIL import Image, ImageOps

import siitec_owl as so
import siitec_ui as su

SAMPLES = ["street_dogs", "street", "taxi", "dog_ball", "cat", "fruit_stall"]
SAMPLE_URL = "https://raw.githubusercontent.com/SIITec-Hackathon-2026/round1/main/assets/playground/"
START_PROMPTS = "a person\na dog\na car\na bicycle"
IDEAS = ["a person", "a dog", "a car", "a bicycle", "a cat", "a tree", "a yellow ball",
         "a traffic light", "a man in a red shirt", "a banana", "a cat's eye", "a dog's tongue"]
PALETTE = [(255, 140, 0), (60, 200, 255), (255, 80, 160), (120, 230, 90), (190, 120, 255),
           (255, 220, 50), (255, 99, 71), (0, 210, 190)]
MAX_SIDE = 1024          # an uploaded picture is shrunk to this before OWLv2 sees it

UPLOAD_ICON = ('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/>'
               '<line x1="12" x2="12" y1="3" y2="15"/>')  # Lucide "upload", ISC licence
su.ICONS.setdefault("upload", UPLOAD_ICON)

STYLE = su.STYLE.replace("</style>", su._icon_css("ic-upload", "upload") + """
.siitec .play{display:flex;gap:0}
.siitec .play .left{width:340px;flex:none;border-right:1px solid var(--line);padding:10px 12px;gap:8px}
.siitec .play .right{flex:1;min-width:0;padding:10px 12px;gap:8px}
.siitec .lbl{font-size:12px;color:var(--muted);letter-spacing:.3px}
.siitec .thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.siitec .thumbs button{height:auto!important;padding:0!important;border:2px solid transparent!important;background:none!important;line-height:0!important}
.siitec .thumbs button.on{border-color:var(--accent)!important}
.siitec .thumbs img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}
.siitec .ideas{display:flex;flex-wrap:wrap;gap:4px}
.siitec .ideas button{font-size:12px!important;height:24px!important;line-height:22px!important;padding:0 8px!important;border-radius:12px!important}
.siitec .prompts textarea{font-family:'JetBrains Mono','Sarabun',monospace!important;font-size:13px!important;background:var(--field)!important;
  color:var(--ink)!important;border:1px solid var(--fieldline)!important;border-radius:2px!important}
.siitec .shotbig img{max-width:100%;max-height:560px;display:block;margin:0 auto}
.siitec .found{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px}
.siitec .found .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.siitec .found .n{color:var(--muted);margin-left:6px}
.siitec .widget-readout{color:var(--ink)!important}
.siitec .upl .widget-upload,.siitec .upl button{width:100%!important}
</style>""")


def _thumb_b64(rgb, size):
    img = Image.fromarray(rgb)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def load_sample(name):
    """A sample picture from assets/playground beside this file, or from the repo on Colab."""
    p = Path(__file__).with_name("assets") / "playground" / f"{name}.jpg"
    data = p.read_bytes() if p.exists() else urllib.request.urlopen(SAMPLE_URL + name + ".jpg", timeout=20).read()
    return _decode(data)


def _decode(data):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(img)


class Play:
    def __init__(self):
        self.finder = so.Finder()
        self.pictures = {n: load_sample(n) for n in SAMPLES}
        self.name = SAMPLES[0]
        self._seen = {}               # picture name -> OWLv2's look at it (patch embeddings, boxes)

    # ------------------------------------------------------------ model
    @torch.no_grad()
    def _look(self, name):
        """The vision half of OWLv2, once per picture: every patch's embedding and its box."""
        if name not in self._seen:
            rgb = self.pictures[name]
            f, m = self.finder, self.finder.model
            px = f.proc(images=[Image.fromarray(rgb)], return_tensors="pt")["pixel_values"].to(f.device, f.dtype)
            fmap = m.image_embedder(pixel_values=px)[0]
            feats = fmap.reshape(1, -1, fmap.shape[-1])
            head = m.class_head
            emb = head.dense0(feats)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
            shift = head.logit_shift(feats)[..., 0]
            scale = head.elu(head.logit_scale(feats))[..., 0] + 1
            cx, cy, w, h = m.box_predictor(feats, fmap).float().unbind(-1)
            side = max(rgb.shape[:2])
            boxes = torch.stack([(cx - w / 2) * side, (cy - h / 2) * side, (cx + w / 2) * side, (cy + h / 2) * side], -1)
            self._seen[name] = (emb[0], shift[0], scale[0], boxes[0])
        return self._seen[name]

    @torch.no_grad()
    def find(self, name, prompts, threshold):
        """Boxes for each prompt scoring `threshold` or more: [(prompt index, score, box)]."""
        emb, shift, scale, boxes = self._look(name)
        q = self.finder.text(prompts)
        s = ((emb @ q.T + shift[:, None]) * scale[:, None]).float().sigmoid()    # patches x prompts
        out = []
        for k in range(len(prompts)):
            v = s[:, k]
            keep = (v >= threshold).nonzero()[:, 0]
            if len(keep):
                keep = keep[torchvision.ops.nms(boxes[keep], v[keep], 0.3)][:30]
                out += [(k, float(v[i]), boxes[i].tolist()) for i in keep]
        return out

    # ------------------------------------------------------------ drawing
    def draw(self, name, prompts, found):
        img = self.pictures[name].copy()
        h, w = img.shape[:2]
        t = max(2, max(h, w) // 350)
        for k, sc, (x1, y1, x2, y2) in sorted(found, key=lambda f: f[1]):
            c = PALETTE[k % len(PALETTE)]
            p1, p2 = (int(max(0, x1)), int(max(0, y1))), (int(min(w - 1, x2)), int(min(h - 1, y2)))
            cv2.rectangle(img, p1, p2, c, t)
            text = f"{prompts[k]} {sc:.2f}"
            fs = t / 4.2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
            y = p1[1] - 4 if p1[1] - th - 8 > 0 else p1[1] + th + 6
            x = max(0, min(p1[0], w - tw - 6))          # the label stays inside the picture
            cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 4), c, -1)
            cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), max(1, t // 2), cv2.LINE_AA)
        return img

    # ------------------------------------------------------------ screen
    def show(self):
        title = W.HBox([su._html(su.HEADER.replace("· รอบคัดเลือก", "· ลองเล่น OWLv2")), W.HBox(layout=W.Layout(flex="1"))],
                       layout=W.Layout(align_items="center"))
        title.add_class("titlebar")
        self.run = su._button("Run", primary=True)
        self.run.add_class("run-btn")
        title.children = title.children + (self.run,)

        self.thumb_buttons = {}
        for n in SAMPLES:
            b = W.Button(layout=W.Layout(width="auto"))
            b.on_click(lambda _, n=n: self._pick(n))
            self.thumb_buttons[n] = b
        self.thumbs = thumbs = W.GridBox(list(self.thumb_buttons.values()))
        thumbs.add_class("thumbs")
        self.thumb_css = su._html()
        self.upload = W.FileUpload(accept="image/*", multiple=False, description="อัปโหลดรูปของเรา", icon="")
        self.upload.add_class("ic-upload")
        self.upload.observe(self._uploaded, "value")
        upl = W.HBox([self.upload])
        upl.add_class("upl")

        self.prompts = W.Textarea(value=START_PROMPTS, layout=W.Layout(width="100%", height="130px"))
        pbox = W.VBox([self.prompts])
        pbox.add_class("prompts")
        ideas = []
        for idea in IDEAS:
            b = W.Button(description=idea, layout=W.Layout(width="auto"))
            b.on_click(lambda _, i=idea: self._add_idea(i))
            ideas.append(b)
        ideas_box = W.HBox(ideas)
        ideas_box.add_class("ideas")
        self.th = W.FloatSlider(value=0.15, min=0.01, max=0.6, step=0.01, readout_format=".2f",
                                layout=W.Layout(width="100%"), continuous_update=False)
        self.th.observe(lambda _: self._press(), "value")

        left = W.VBox([su._html('<div class="lbl">1. เลือกรูป</div>'), thumbs, upl,
                       su._html('<div class="lbl">2. พิมพ์สิ่งที่อยากให้หา (ภาษาอังกฤษ บรรทัดละอย่าง)</div>'), pbox,
                       su._html('<div class="lbl">ลองคำเหล่านี้ (กดเพื่อเพิ่ม)</div>'), ideas_box,
                       su._html('<div class="lbl">3. ความมั่นใจขั้นต่ำ — ต่ำ = เจอเยอะแต่ผิดเยอะ · สูง = เจอน้อยแต่แม่น</div>'), self.th])
        left.add_class("left")
        self.picture = su._html()
        self.picture.add_class("shotbig")
        self.found = su._html()
        right = W.VBox([self.picture, self.found])
        right.add_class("right")
        body = W.HBox([left, right])
        body.add_class("play")
        self.statusbar = su._html('<div class="statusbar">พร้อม</div>')
        self.run.on_click(lambda _: self._press())

        box = W.VBox([su._html(STYLE), self.thumb_css, title, body, self.statusbar])
        box.add_class("siitec")
        display(box)
        self._fill_thumbs()
        self._press()

    def _fill_thumbs(self):
        for n, b in self.thumb_buttons.items():
            b.description = ""
            b.tooltip = n
            (b.add_class if n == self.name else b.remove_class)("on")
        # a Button cannot hold an image, so each thumbnail is a background picture set by a per-button class
        css = "".join(f".siitec .thumbs button.t-{n}{{background:url(data:image/jpeg;base64,{_thumb_b64(self.pictures[n], 200)}) center/cover!important;aspect-ratio:4/3}}"
                      for n in self.thumb_buttons)
        for n, b in self.thumb_buttons.items():
            b.add_class(f"t-{n}")
        self.thumb_css.value = f"<style>{css}</style>"

    def _pick(self, name):
        self.name = name
        for n, b in self.thumb_buttons.items():
            (b.add_class if n == name else b.remove_class)("on")
        self._press()

    def _add_idea(self, idea):
        lines = [l for l in self.prompts.value.splitlines() if l.strip()]
        if idea not in lines:
            self.prompts.value = "\n".join(lines + [idea])
        self._press()

    def _uploaded(self, ch):
        v = ch["new"]
        if not v:
            return
        item = v[0] if isinstance(v, (list, tuple)) else next(iter(v.values()))     # ipywidgets 8 / 7
        try:
            rgb = _decode(bytes(item["content"]))
        except Exception:
            self.statusbar.value = f'<div class="statusbar bad">{su._ic("circle-x")} เปิดไฟล์นี้เป็นรูปไม่ได้ — ลองไฟล์ .jpg หรือ .png</div>'
            return
        name = "upload"
        self.pictures[name] = rgb
        self._seen.pop(name, None)
        if name not in self.thumb_buttons:
            b = W.Button(layout=W.Layout(width="auto"))
            b.on_click(lambda _: self._pick(name))
            self.thumb_buttons[name] = b
            self.thumbs.children = self.thumbs.children + (b,)
        self._fill_thumbs()
        self._pick(name)

    def _press(self):
        prompts = [p.strip() for p in self.prompts.value.splitlines() if p.strip()][:len(PALETTE) * 2]
        if not prompts:
            self.statusbar.value = f'<div class="statusbar bad">{su._ic("triangle-alert")} พิมพ์อย่างน้อย 1 คำ เช่น a dog</div>'
            return
        self.run.disabled = True
        self.statusbar.value = f'<div class="statusbar">{su._ic("loader-circle", "spin")} กำลังหา…</div>'
        try:
            found = self.find(self.name, prompts, self.th.value)
            img = self.draw(self.name, prompts, found)
            self.picture.value = f'<img src="data:image/jpeg;base64,{_thumb_b64(img, 1024)}">'
            rows = []
            for k, p in enumerate(prompts):
                hits = [sc for kk, sc, _ in found if kk == k]
                c = PALETTE[k % len(PALETTE)]
                best = f" · สูงสุด {max(hits):.2f}" if hits else ""
                rows.append(f'<span><span class="dot" style="background:rgb{c}"></span>{html.escape(p)}'
                            f'<span class="n">{len(hits)} กรอบ{best}</span></span>')
            self.found.value = f'<div class="found">{"".join(rows)}</div>'
            ascii_only = all(p.isascii() for p in prompts)
            note = "" if ascii_only else f' · {su._ic("triangle-alert")} OWLv2 อ่านภาษาอังกฤษเท่านั้น'
            self.statusbar.value = (f'<div class="statusbar"><span>{su._ic("circle-check")} เจอ {len(found)} กรอบ</span>'
                                    f'<span>ความมั่นใจ ≥ {self.th.value:.2f}{note}</span></div>')
        except Exception as e:
            self.statusbar.value = f'<div class="statusbar bad">{su._ic("circle-x")} {html.escape(str(e))}</div>'
        finally:
            self.run.disabled = False
