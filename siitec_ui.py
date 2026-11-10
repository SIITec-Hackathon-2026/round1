"""SIITec Hackathon 2569, round 1: the notebook's screen (ipywidgets).

App() gets everything ready (GPU, data, model, picture features) and app.show()
draws the screen. The team's script (my_filter, my_draw) is typed in the Script tab
and run again each time ▶ is pressed.
"""
import base64
import html
import io
import json
import time
import traceback
from pathlib import Path

import cv2
import ipywidgets as W
import numpy as np
import torch
from IPython.display import HTML, display
from PIL import Image

import siitec_owl as so

AUTOSAVE = "siitec_autosave.json"
PER_PAGE = 12
START_THRESHOLD = 0.03
COLORS = {  # class -> RGB, readable on the pictures
    "bottle_broken_large": (255, 99, 71), "bottle_broken_small": (255, 170, 60),
    "bottle_contamination": (190, 120, 255), "metal_nut_bent": (255, 80, 160),
    "metal_nut_color": (60, 200, 255), "metal_nut_flip": (255, 220, 50),
    "metal_nut_scratch": (120, 255, 120), "screw_defect": (255, 120, 60),
}

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
.siitec{--bg:#ffffff;--card:#fff;--field:#f6f8fb;--ink:#14202e;--muted:#6b7a8c;--line:#e4e9f0;--accent:#0891b2;--on-accent:#fff;
  --good:#15803d;--bad:#dc2626;--warn:#b45309}
@media (prefers-color-scheme: dark){.siitec{--bg:#0e1624;--card:#131c2b;--field:#0f1724;--ink:#e6edf7;--muted:#8b9bb0;--line:#243246;
  --accent:#38d6f0;--on-accent:#04212b;--good:#34d399;--bad:#f87171;--warn:#fbbf24}}
.siitec{color:var(--ink);background:var(--bg);border-radius:14px;padding:14px 18px;box-sizing:border-box;font-family:'Sarabun',system-ui,sans-serif;font-size:15px;line-height:1.45;max-width:1100px;
  --jp-widgets-color:var(--ink);--jp-widgets-label-color:var(--ink);--jp-widgets-readout-color:var(--muted);
  --jp-widgets-input-color:var(--ink);--jp-widgets-input-background-color:var(--field);--jp-widgets-input-border-color:var(--line);
  --jp-widgets-input-focus-border-color:var(--accent);--jp-widgets-slider-handle-background-color:var(--accent);
  --jp-widgets-slider-active-handle-color:var(--accent);--jp-widgets-slider-handle-border-color:var(--accent);
  --jp-widgets-font-size:15px;--jp-layout-color1:var(--card);--jp-border-color1:var(--line)}
.siitec *{font-family:inherit}
.siitec .widget-label,.siitec label{color:var(--ink)!important}
.siitec textarea,.siitec input[type=text],.siitec select{background:var(--field)!important;color:var(--ink)!important;
  border:1px solid var(--line)!important;border-radius:8px!important;font-size:14px!important}
.siitec textarea{font-family:'JetBrains Mono',ui-monospace,monospace!important;font-size:13px!important;padding:6px 9px!important;line-height:1.5!important;resize:vertical}
.siitec .siitec-code textarea{tab-size:4;white-space:pre;overflow:auto;padding:10px 12px!important}
.siitec textarea:focus,.siitec input:focus,.siitec select:focus{border-color:var(--accent)!important;outline:none}
.siitec .widget-readout{color:var(--muted)!important;font-family:'JetBrains Mono',monospace;font-size:12.5px}
.siitec .noUi-target,.siitec .ui-slider{background:var(--line)!important;border:none!important;box-shadow:none!important}
.siitec .noUi-handle,.siitec .ui-slider .ui-slider-handle{background:var(--accent)!important;border:2px solid var(--card)!important;box-shadow:none!important}
.siitec button,.siitec .jupyter-button{border-radius:8px!important;font-weight:600!important;border:1px solid var(--line)!important;
  background:transparent!important;color:var(--ink)!important;height:34px;padding:0 14px!important;box-shadow:none!important}
.siitec button:hover{border-color:var(--accent)!important}
.siitec .primary button,.siitec button.primary{background:var(--accent)!important;color:var(--on-accent)!important;border-color:var(--accent)!important;height:40px;padding:0 20px!important;font-size:15px!important}
.siitec .lm-TabBar-tab,.siitec .p-TabBar-tab{background:transparent!important;color:var(--muted)!important;border:none!important;
  border-bottom:2px solid transparent!important;font-size:15px!important;font-weight:600;padding:6px 14px!important;min-width:auto!important;flex:0 0 auto!important}
.siitec .lm-TabBar-tab.lm-mod-current,.siitec .p-TabBar-tab.p-mod-current{color:var(--ink)!important;border-bottom-color:var(--accent)!important}
.siitec .lm-TabBar,.siitec .p-TabBar{border-bottom:1px solid var(--line)!important;background:transparent!important}
.siitec .lm-TabBar-content,.siitec .p-TabBar-content{border:none!important}
.siitec .widget-tab-contents{background:transparent!important;border:none!important;padding:12px 0 0!important}
.siitec .widget-checkbox input{accent-color:var(--accent)}
.siitec .hdr{display:flex;align-items:center;gap:10px;padding:4px 0 2px}
.siitec .logo{height:28px;display:flex;align-items:center;font-weight:700;font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:0 7px}
.siitec .logo img{height:28px}
.siitec .t1{font-size:18px;font-weight:700} .siitec .t2{color:var(--muted);font-size:14px}
.siitec .score{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:var(--accent);line-height:1}
.siitec .k{font-size:12.5px;color:var(--muted)}
.siitec .up{color:var(--good)} .siitec .down{color:var(--bad)} .siitec .muted{color:var(--muted)}
.siitec .status{font-size:13px;color:var(--muted)}
.siitec .part{font-size:12.5px;font-weight:700;color:var(--muted);letter-spacing:.4px;text-transform:uppercase;margin:10px 0 0}
.siitec .row{border-bottom:1px solid var(--line);padding:6px 0}
.siitec .name{display:flex;gap:8px;align-items:flex-start;line-height:1.3}
.siitec .dot{width:9px;height:9px;border-radius:50%;flex:none;margin-top:6px}
.siitec .ckey{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted)}
.siitec .f1{font-family:'JetBrains Mono',monospace;text-align:right;line-height:1.25;white-space:nowrap}
.siitec .f1 b{font-size:16px} .siitec .f1 span{font-size:11px;color:var(--muted)}
.siitec .gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.siitec .shot{border-radius:8px;overflow:hidden;outline:none;border:1px solid var(--line);transition:transform .15s;background:var(--card)}
.siitec .shot img{width:100%;display:block;aspect-ratio:1/1;object-fit:cover;background:#000}
.siitec .shot .cap{display:flex;gap:6px;align-items:center;padding:4px 7px;font-size:12px}
.siitec .shot .fn{font-family:'JetBrains Mono',monospace;color:var(--muted);margin-right:auto;font-size:11px}
.siitec .ok{color:var(--good);font-weight:600} .siitec .miss{color:var(--bad);font-weight:600} .siitec .fa{color:var(--warn);font-weight:600}
.siitec .shot:focus{transform:scale(2.2);transform-origin:center top;z-index:20;position:relative;box-shadow:0 10px 40px rgba(0,0,0,.45)}
.siitec .err{color:var(--bad);font-weight:600}
.siitec .one{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;align-items:start}
.siitec .one img{width:100%;border-radius:8px;display:block}
@media (max-width:760px){.siitec .one{grid-template-columns:1fr}}
.siitec table.dbg{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px}
.siitec table.dbg th{color:var(--muted);font-weight:600;text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);font-size:12px}
.siitec table.dbg td{padding:4px 6px;border-bottom:1px solid var(--line)}
.siitec .num{font-family:'JetBrains Mono',monospace;text-align:right;white-space:nowrap}
.siitec .widget-toggle-buttons .widget-toggle-button{height:34px}
.siitec .widget-toggle-buttons button.mod-active{background:var(--accent)!important;color:var(--on-accent)!important;border-color:var(--accent)!important}
.siitec .errbox{white-space:pre-wrap;font-family:'JetBrains Mono',monospace;font-size:12px;border-left:3px solid var(--bad);padding:6px 10px;margin-top:4px}
.siitec code{font-family:'JetBrains Mono',monospace;font-size:12.5px;background:var(--field);padding:1px 5px;border-radius:5px}
.siitec .prog{height:4px;border-radius:4px;background:var(--line);overflow:hidden;max-width:360px;margin:4px 0 0 26px}
.siitec .prog i{display:block;height:100%;background:var(--accent)}
</style>"""


def _logo(name, text):
    p = Path(__file__).with_name("assets") / name
    if p.exists():
        mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
        return f'<span class="logo"><img src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"></span>'
    return f'<span class="logo">{text}</span>'


HEADER = (f'<div class="hdr">{_logo("kmitl.png", "KMITL")}{_logo("siitec.png", "SIITec")}'
          '<span class="t1">SIITec Hackathon 2569</span><span class="t2">· รอบคัดเลือก</span></div>')


def _html(s=""):
    return W.HTML(s)


def _button(text, primary=False):
    b = W.Button(description=text, layout=W.Layout(width="auto"))
    if primary:
        b.add_class("primary")
    return b


SCRIPT_TEMPLATE = '''import numpy as np
import cv2

# เกณฑ์ความมั่นใจของแต่ละแบบ (0 ถึง 1) — กรอบที่คะแนนต่ำกว่านี้ถูกตัดทิ้ง
THRESHOLD = {
    "bottle_broken_large":  0.03,
    "bottle_broken_small":  0.03,
    "bottle_contamination": 0.03,
    "metal_nut_bent":       0.03,
    "metal_nut_color":      0.03,
    "metal_nut_flip":       0.03,
    "metal_nut_scratch":    0.03,
    "screw_defect":         0.03,
}


def my_filter(image, detections):
    """เลือกกรอบที่จะเก็บไว้
    image      : ภาพ (numpy RGB  สูง x กว้าง x 3)
    detections : [{"label": "metal_nut_bent", "score": 0.31, "box": [x1, y1, x2, y2]}, ...]
    """
    h, w = image.shape[:2]
    keep = []
    for d in detections:
        if d["score"] < THRESHOLD[d["label"]]:
            continue
        # ไอเดีย: ตัดกรอบเล็ก/ใหญ่ผิดปกติ · ตัดกรอบนอกตัวชิ้นงาน · ภาพละไม่เกินกี่กรอบ
        keep.append(d)
    return keep


def my_draw(image, found, answers):
    """วาดผลลัพธ์ด้วย OpenCV แล้วคืนภาพ (RGB)
    found : กรอบที่เก็บไว้    answers : คำตอบ (มีเฉพาะชุดฝึก)
    """
    img = image.copy()
    t = max(2, img.shape[0] // 300)                  # ความหนาเส้นตามขนาดภาพ
    for a in answers:                                # คำตอบ: เส้นขาวทับเส้นเขียว
        x1, y1, x2, y2 = map(int, a["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), t + 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 200, 120), t)
    for d in found:                                  # ที่หาเจอ: กรอบส้ม + ชื่อ + คะแนน
        x1, y1, x2, y2 = map(int, d["box"])
        color = (255, 140, 0)                        # (R, G, B)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t + 1)
        text = d["label"].split("_")[-1] + f' {d["score"]:.2f}'
        cv2.putText(img, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    t / 3, color, max(1, t // 2), cv2.LINE_AA)
    return img
'''

# Tab in the script box types four spaces instead of leaving the box
TAB_KEY = """
<script>
(function(){
  function hook(root){
    root.querySelectorAll('.siitec-code textarea:not([data-tab])').forEach(function(t){
      t.dataset.tab = 1;
      t.addEventListener('keydown', function(e){
        if (e.key !== 'Tab') return;
        e.preventDefault();
        var s = t.selectionStart, v = t.value;
        t.value = v.slice(0, s) + '    ' + v.slice(t.selectionEnd);
        t.selectionStart = t.selectionEnd = s + 4;
        t.dispatchEvent(new Event('input', {bubbles: true}));
      });
    });
  }
  new MutationObserver(function(){ hook(document); }).observe(document.body, {childList: true, subtree: true});
  hook(document);
})();
</script>
"""


def _script_line(e):
    """Line number inside the team's script for an error raised there (or None)."""
    if isinstance(e, SyntaxError) and e.filename == "<script>":
        return e.lineno
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == "<script>"]
    return lines[-1] if lines else None


def _err_text(e):
    line = _script_line(e)
    msg = e.msg if isinstance(e, SyntaxError) else str(e)
    return f"{type(e).__name__}: {msg}" + (f"  (บรรทัด {line})" if line else "")


def _root(children, gap="10px"):
    box = W.VBox([_html(STYLE)] + children, layout=W.Layout(grid_gap=gap))
    box.add_class("siitec")
    return box


def default_draw(image, found, answers):
    """Built-in drawing, used when the notebook has no my_draw (or it fails)."""
    img = image.copy()
    t = max(2, img.shape[0] // 300)
    for a in answers:
        x1, y1, x2, y2 = map(int, a["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), t + 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 200, 120), t)
    for d in found:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(d["label"], (255, 140, 0)), t + 1)
    return img


def _jpeg_b64(rgb, size=560):
    img = Image.fromarray(rgb)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=84)
    return base64.b64encode(buf.getvalue()).decode()


class Dataset:
    """A picture set with its features, the latest boxes and their marks."""

    def __init__(self, name, finder, folder, progress=None):
        self.name = name
        self.dir, self.records, self.classes = so.load_set(name, folder)
        self.cache = finder.embed_pictures(self.dir, self.records, progress=progress)
        self.candidates = self.found = None
        self.score = self.per = self.marks = None


class App:
    def __init__(self, folder="."):
        self.folder = folder
        box = _html()
        self.setup_view = _root([box])
        display(self.setup_view)
        lines = []

        def show(frac=None):
            bar = f'<div class="prog"><i style="width:{frac * 100:.0f}%"></i></div>' if frac is not None else ""
            box.value = "<br>".join(lines) + bar

        def step(label, fn):
            lines.append(f"⏳ {label}")
            show()
            try:
                v = fn()
            except Exception as e:
                lines[-1] = f'❌ {label} — <span class="err">{html.escape(str(e))}</span>'
                show()
                raise
            lines[-1] = f"✅ {label}"
            show()
            return v

        if torch.cuda.is_available():
            step(f"GPU {torch.cuda.get_device_name(0)}", lambda: None)
        else:
            lines.append('⚠️ ไม่มี GPU — ช้ามาก · Colab: <b>Runtime ▸ Change runtime type ▸ T4 GPU</b>')
        self.finder = step("โหลดโมเดล OWLv2", so.Finder)
        self.train = step("เตรียมภาพชุดฝึก (ครั้งแรกประมาณ 1 นาที)",
                          lambda: Dataset("train", self.finder, folder, progress=lambda i, n: show(i / n)))
        lines.append("พร้อมแล้ว")
        show()
        self.test = None
        self.has_test = so.test_available(folder)
        self.classes = self.train.classes
        saved = self._load_autosave()
        self.prompts = saved.get("prompts") or {c: [self._default_prompt(c)] for c in self.classes}
        self.team = saved.get("team", "")
        self.script = saved.get("script") or SCRIPT_TEMPLATE
        self.fns, self.script_error, self.filter_error = {}, None, None
        self.best = None

    # ------------------------------------------------------------ state
    @staticmethod
    def _default_prompt(c):
        part = so.CLASS_INFO[c][0]
        return f"{c[len(part) + 1:].replace('_', ' ')} on a {part.replace('_', ' ')}"

    def _load_autosave(self):
        try:
            return json.loads(Path(AUTOSAVE).read_text())
        except Exception:
            return {}

    def _autosave(self):
        try:
            Path(AUTOSAVE).write_text(json.dumps({"team": self.team_box.value, "prompts": self.prompts,
                                                  "script": self.script}, ensure_ascii=False, indent=1))
        except Exception:
            pass

    def _read_inputs(self):
        for c in self.classes:
            self.prompts[c] = [l for l in self.areas[c].value.split("\n") if l.strip()]
        self.script = self.script_area.value
        self.fns, self.script_error = {}, None
        try:
            ns = {"np": np, "cv2": cv2}
            exec(compile(self.script, "<script>", "exec"), ns)
            self.fns = {k: ns[k] for k in ("my_filter", "my_draw") if callable(ns.get(k))}
        except Exception as e:
            self.script_error = _err_text(e)

    # ------------------------------------------------------------ running
    def _run(self, ds, find=True):
        """find=True: prompts changed (text part again); False: my_filter only."""
        self._read_inputs()
        if find or ds.candidates is None:
            ds.candidates = self.finder.find(ds.cache, ds.records, self.prompts)
        # without a (working) my_filter every class keeps boxes scoring START_THRESHOLD or more
        plain = so.apply_thresholds(ds.candidates, {c: START_THRESHOLD for c in self.classes})
        found, self.filter_error = plain, None
        fn = self.fns.get("my_filter")
        if fn:
            try:
                found = so.apply_filter(fn, ds.dir, ds.records, ds.candidates)
            except Exception as e:
                self.filter_error = _err_text(e)
        ds.found = found
        ds.score, ds.per, ds.marks = so.evaluate(ds.records, found, ds.classes)

    def _press(self, find):
        t = time.time()
        self.status.value = '<span class="status">⏳</span>'
        try:
            prev = self.train.score
            self._run(self.train, find)
            self._autosave()
            self._show_score(prev, time.time() - t)
            for c in self.classes:
                v = self.train.per[c]
                self.f1s[c].value = (f'<div class="f1"><b>{v["f1"]:.2f}</b><br>'
                                     f'<span>✓{v["tp"]} ✗{v["fp"]} ?{v["fn"]}</span></div>')
            self.gallery.refresh()
        except Exception as e:
            self.status.value = f'<span class="err">{html.escape(type(e).__name__ + ": " + str(e))}</span>'

    def _show_score(self, prev, secs):
        s = self.train.score
        self.best = s if self.best is None else max(self.best, s)
        delta = ""
        if prev is not None and abs(s - prev) > 1e-6:
            delta = f'<span class="{"up" if s > prev else "down"}">{"▲" if s > prev else "▼"}{abs(s - prev):.3f}</span>'
        self.score_html.value = (f'<div class="k">คะแนนชุดฝึก</div><div class="score">{s:.3f}</div>'
                                 f'<div class="k">{delta} ดีสุด {self.best:.3f}</div>')
        flt = "my_filter ✓" if self.fns.get("my_filter") else f"ไม่มี my_filter (ใช้เกณฑ์ {START_THRESHOLD})"
        msg, err = f'<span class="status">{secs:.1f} วินาที · {flt}</span>', ""
        if self.script_error:
            err = f'<span class="err">Script ผิดพลาด — ใช้เกณฑ์ {START_THRESHOLD} แทน</span><div class="errbox">{html.escape(self.script_error)}</div>'
        elif self.filter_error:
            err = f'<span class="err">my_filter ผิดพลาด — ใช้เกณฑ์ {START_THRESHOLD} แทน</span><div class="errbox">{html.escape(self.filter_error)}</div>'
        self.status.value = err or msg
        self.script_msg.value = err or '<span class="ok">✓ Script ทำงานได้</span>' 

    # ------------------------------------------------------------ screen
    def show(self):
        run = _button("▶ ทดสอบ", primary=True)
        run.on_click(lambda _: self._press(True))
        self.score_html = _html('<div class="k">คะแนนชุดฝึก</div><div class="score">–</div><div class="k">&nbsp;</div>')
        self.status = _html('<span class="status">แก้คำค้นแล้วกด ▶</span>')
        top = W.HBox([self.score_html, run, self.status], layout=W.Layout(align_items="center", grid_gap="18px"))

        self.gallery = Gallery(lambda: self.train, lambda: self.fns.get("my_draw"))
        pages = [self._prompt_page(), self._script_page(), self.gallery.box, self._export_page()]
        titles = ["คำค้น", "Script", "ภาพ", "ส่งงาน"]
        if self.has_test:
            pages.append(self._test_page())
            titles.append("🔓 ชุดทดสอบ")
        tab = W.Tab(children=pages)
        for i, t in enumerate(titles):
            tab.set_title(i, t)
        self.tab = tab
        self.setup_view.layout.display = "none"     # the ready list has done its job
        display(_root([_html(HEADER), top, tab]))
        display(HTML(TAB_KEY))
        self._press(True)

    def _prompt_page(self):
        self.areas, self.f1s = {}, {}
        rows = [_html('<div class="k">คำค้นภาษาอังกฤษ บรรทัดละแบบ · ขึ้นต้นด้วย <code>-</code> = “ไม่ใช่สิ่งนี้” · '
                      'เกณฑ์ความมั่นใจตั้งในแท็บ Script</div>')]
        for part in ("bottle", "metal_nut", "screw"):
            rows.append(_html(f'<div class="part">{so.PART_TH[part]}</div>'))
            for c in [c for c in self.classes if so.CLASS_INFO[c][0] == part]:
                name = _html(f'<div class="name"><span class="dot" style="background:rgb{COLORS[c]}"></span>'
                             f'<span>{so.CLASS_INFO[c][1]}<br><span class="ckey">{c}</span></span></div>')
                name.layout = W.Layout(width="230px", flex="none")
                area = W.Textarea(value="\n".join(self.prompts.get(c, [])), placeholder="a scratch on a metal nut",
                                  layout=W.Layout(flex="1 1 auto", height="54px", width="auto"))
                f1 = _html('<div class="f1"><b>–</b></div>')
                f1.layout = W.Layout(width="104px", flex="none")
                self.areas[c], self.f1s[c] = area, f1
                row = W.HBox([name, area, f1], layout=W.Layout(align_items="center", grid_gap="12px"))
                row.add_class("row")
                rows.append(row)
        rows.append(_html('<div class="k" style="margin-top:6px">F1 ของแต่ละแบบ · ✓ ถูก ✗ ทักผิด ? พลาด</div>'))
        return W.VBox(rows)

    def _script_page(self):
        self.script_area = W.Textarea(value=self.script, layout=W.Layout(width="100%", height="560px"))
        self.script_area.add_class("siitec-code")
        self.script_msg = _html()
        reset = _button("ใช้โค้ดตั้งต้น")

        def do_reset(_):
            if reset.description != "กดอีกครั้งเพื่อยืนยัน":
                reset.description = "กดอีกครั้งเพื่อยืนยัน"
                return
            reset.description = "ใช้โค้ดตั้งต้น"
            self.script_area.value = SCRIPT_TEMPLATE
            self._press(False)

        reset.on_click(do_reset)
        return W.VBox([
            _html('<div class="k">Python + NumPy + OpenCV · <code>my_filter</code> เลือกกรอบ · <code>my_draw</code> วาดภาพในแท็บภาพ · '
                  'แก้แล้วกด ▶ ทดสอบ · ห้ามเทรนหรือใช้โมเดลอื่น</div>'),
            self.script_area,
            W.HBox([self.script_msg, reset], layout=W.Layout(justify_content="space-between", align_items="center")),
        ], layout=W.Layout(grid_gap="8px"))

    def _export_page(self):
        self.team_box = W.Text(value=self.team, placeholder="ชื่อทีม", layout=W.Layout(width="280px"))
        save = _button("ส่งออกไฟล์ส่งงาน", primary=True)
        msg = _html()
        up = W.FileUpload(accept=".json", multiple=False, description="เปิดไฟล์เดิม", icon="")
        up_msg = _html()

        def do_save(_):
            team = self.team_box.value.strip()
            if not team:
                msg.value = '<span class="err">ใส่ชื่อทีมก่อน</span>'
                return
            self._read_inputs()
            exp = so.make_export(team, self.prompts, self.script, round(self.train.score or 0, 4))
            safe = "".join(ch if ch.isalnum() else "_" for ch in team)[:40] or "team"
            path = Path(f"siitec_round1_{safe}.json")
            path.write_text(json.dumps(exp, ensure_ascii=False, indent=1))
            self._autosave()
            msg.value = f'<span class="ok">✓ {path.name}</span> <span class="muted">คะแนนชุดฝึก {exp["train_score"]:.3f}</span>'
            try:
                from google.colab import files
                files.download(str(path))
            except Exception:
                msg.value += f' · <a href="{path.name}" download style="color:var(--accent)">ดาวน์โหลด</a>'

        def do_load(change):
            files = change["new"]
            item = files[0] if isinstance(files, (list, tuple)) else next(iter(files.values()), None)
            if not item:
                return
            try:
                data = json.loads(bytes(item["content"]).decode("utf-8"))
                for c in self.classes:
                    if c in data.get("prompts", {}):
                        self.areas[c].value = "\n".join(data["prompts"][c])
                if data.get("script"):
                    self.script_area.value = data["script"]
                self.team_box.value = data.get("team", self.team_box.value)
                up_msg.value = '<span class="ok">✓ ได้คำค้นและ Script คืนแล้ว</span>'
                self._press(True)
            except Exception as e:
                up_msg.value = f'<span class="err">เปิดไม่ได้: {html.escape(str(e))}</span>'

        save.on_click(do_save)
        up.observe(do_load, "value")
        return W.VBox([
            _html('<div class="k">ไฟล์มีคำค้นและ Script — ส่งพร้อมวิดีโอ · คะแนนจริงมาจากชุดทดสอบลับ</div>'),
            W.HBox([self.team_box, save], layout=W.Layout(grid_gap="10px", align_items="center")), msg,
            W.HBox([up, up_msg], layout=W.Layout(grid_gap="10px", align_items="center", margin="14px 0 0")),
        ], layout=W.Layout(grid_gap="8px"))

    def _test_page(self):
        go = _button("▶ รันกับชุดทดสอบ", primary=True)
        msg = _html('<span class="k">ชุดทดสอบเปิดแล้ว — ลองกับภาพที่ไม่เคยเห็น</span>')
        holder = W.VBox()

        def run(_):
            try:
                if self.test is None:
                    msg.value = '<span class="status">⏳ กำลังเตรียมชุดทดสอบ…</span>'
                    self.test = Dataset("test", self.finder, self.folder)
                    self.test_gallery = Gallery(lambda: self.test, lambda: self.fns.get("my_draw"))
                    holder.children = [self.test_gallery.box]
                self._run(self.test, True)
                msg.value = (f'<div class="k">คะแนนชุดทดสอบ</div><div class="score">{self.test.score:.3f}</div>'
                             f'<div class="k">ชุดฝึก {self.train.score:.3f}</div>')
                self.test_gallery.refresh()
            except Exception as e:
                msg.value = f'<span class="err">{html.escape(str(e))}</span>'

        go.on_click(run)
        return W.VBox([W.HBox([go, msg], layout=W.Layout(grid_gap="18px", align_items="center")), holder],
                      layout=W.Layout(grid_gap="10px"))


class Gallery:
    """Pictures drawn with the team's my_draw: a page of them, or one at a time with every candidate box."""

    def __init__(self, get_ds, get_draw):
        self.get_ds, self.get_draw, self.page, self.one = get_ds, get_draw, 0, 0
        self.mode = W.ToggleButtons(options=[("หลายภาพ", "grid"), ("ทีละภาพ", "one")], value="grid")
        self.mode.style.button_width = "104px"
        self.pick = W.Dropdown(options=[("ทุกภาพ", "all"), ("มีที่พลาด", "miss"), ("มีที่ทักผิด", "fa"), ("ถูกหมด", "ok"),
                                        ("ขวด", "bottle"), ("น็อต", "metal_nut"), ("สกรู", "screw")],
                               layout=W.Layout(width="150px"))
        self.prev, self.next = _button("◀"), _button("▶")
        self.where, self.gal, self.msg = _html(), _html(), _html()
        self.mode.observe(lambda _: self.refresh(), "value")
        self.pick.observe(lambda _: self._go(0), "value")
        self.prev.on_click(lambda _: self._go(-1, step=True))
        self.next.on_click(lambda _: self._go(1, step=True))
        bar = W.HBox([self.mode, self.pick, self.prev, self.where, self.next],
                     layout=W.Layout(grid_gap="8px", align_items="center", flex_flow="row wrap"))
        self.foot = _html()
        self.box = W.VBox([bar, self.msg, self.gal, self.foot], layout=W.Layout(grid_gap="8px"))

    def _go(self, n, step=False):
        if self.mode.value == "grid":
            self.page = self.page + n if step else n
        else:
            self.one = self.one + n if step else n
        self.refresh()

    def _picked(self, ds):
        want, pick = self.pick.value, []
        for i, (r, (f_hit, a_hit)) in enumerate(zip(ds.records, ds.marks)):
            miss, fa = a_hit.count(False), f_hit.count(False)
            if (want == "miss" and not miss) or (want == "fa" and not fa) or (want == "ok" and (miss or fa)) \
                    or (want in so.PART_TH and r["category"] != want):
                continue
            pick.append(i)
        return pick

    def _drawn(self, ds, i, img):
        draw, out = self.get_draw(), None
        if draw:
            try:
                out = np.asarray(draw(img.copy(), [dict(d) for d in ds.found[i]],
                                      [dict(a) for a in ds.records[i]["boxes"]]), np.uint8)
            except Exception as e:
                self.msg.value = (f'<span class="err">my_draw ผิดพลาด — ใช้แบบตั้งต้น</span>'
                                  f'<div class="errbox">{html.escape(_err_text(e))}</div>')
        if out is None or out.ndim != 3 or out.shape[:2] != img.shape[:2]:
            out = default_draw(img, ds.found[i], ds.records[i]["boxes"])
        return out

    def refresh(self):
        ds = self.get_ds()
        if ds.marks is None:
            return
        self.msg.value = ""
        pick = self._picked(ds)
        if self.mode.value == "one":
            return self._show_one(ds, pick)
        pages = max(1, -(-len(pick) // PER_PAGE))
        self.page = min(max(self.page, 0), pages - 1)
        self.where.value = f'<span class="k">{self.page + 1}/{pages} · {len(pick)} ภาพ</span>'
        shots = []
        for i in pick[self.page * PER_PAGE:(self.page + 1) * PER_PAGE]:
            r, (f_hit, a_hit) = ds.records[i], ds.marks[i]
            out = self._drawn(ds, i, so.read_rgb(Path(ds.dir) / r["file"]))
            ok, miss, fa = f_hit.count(True), a_hit.count(False), f_hit.count(False)
            marks = (f'<span class="ok">✓{ok}</span>' if ok else "") + (f'<span class="miss">?{miss}</span>' if miss else "") + \
                    (f'<span class="fa">✗{fa}</span>' if fa else "")
            shots.append(f'<div class="shot" tabindex="0"><img src="data:image/jpeg;base64,{_jpeg_b64(out)}">'
                         f'<div class="cap"><span class="fn">{r["file"][:-4]}</span>{marks or "<span class=ok>ปกติ</span>"}</div></div>')
        self.gal.value = f'<div class="gal">{"".join(shots)}</div>' if shots else '<span class="k">ไม่มีภาพ</span>'
        self.foot.value = '<div class="k">วาดด้วย <code>my_draw</code> · คลิกภาพเพื่อขยาย · ดูรายละเอียดทุกกรอบที่ “ทีละภาพ”</div>'

    def _show_one(self, ds, pick):
        """One picture: my_draw's drawing with every box numbered, and a table of the candidates."""
        if not pick:
            self.where.value, self.gal.value, self.foot.value = "", '<span class="k">ไม่มีภาพ</span>', ""
            return
        self.one = min(max(self.one, 0), len(pick) - 1)
        i = pick[self.one]
        r, (f_hit, a_hit) = ds.records[i], ds.marks[i]
        self.where.value = f'<span class="k">{self.one + 1}/{len(pick)} · {r["file"][:-4]}</span>'
        img = so.read_rgb(Path(ds.dir) / r["file"])
        out = self._drawn(ds, i, img)

        kept = {(d["label"], tuple(d["box"])): h for d, h in zip(ds.found[i], f_hit)}
        rows = []  # (label, score, box, kept?, hit?)
        for d in ds.found[i]:
            rows.append((d["label"], d.get("score", 0.0), d["box"], True, kept[(d["label"], tuple(d["box"]))]))
        dropped = [d for d in ds.candidates[i] if (d["label"], tuple(d["box"])) not in kept]
        for d in sorted(dropped, key=lambda d: -d["score"])[:max(0, 12 - len(rows))]:
            would = any(a["label"] == d["label"] and so._iou(d["box"], a["box"]) >= so.IOU_HIT for a in r["boxes"])
            rows.append((d["label"], d["score"], d["box"], False, would))
        rows.sort(key=lambda x: -x[1])

        t = max(2, img.shape[0] // 300)
        for n, (label, score, box, keep, hit) in enumerate(rows, 1):
            x1, y1, x2, y2 = map(int, box)
            if not keep:
                cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), max(1, t // 2))
            tag = str(n)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, t / 3, max(1, t // 2))
            cv2.rectangle(out, (x2 - tw - 6, y1), (x2, y1 + th + 6), (20, 20, 20), -1)
            cv2.putText(out, tag, (x2 - tw - 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, t / 3, (255, 255, 255),
                        max(1, t // 2), cv2.LINE_AA)

        trs = []
        for n, (label, score, box, keep, hit) in enumerate(rows, 1):
            if keep:
                res = '<span class="ok">ถูก</span>' if hit else '<span class="fa">ทักผิด</span>'
            else:
                res = '<span class="miss">ตัดตัวถูกทิ้ง</span>' if hit else '<span class="muted">ตัดแล้ว ดี</span>'
            w, h = box[2] - box[0], box[3] - box[1]
            trs.append(f'<tr style="{"" if keep else "opacity:.6"}"><td class="num">{n}</td>'
                       f'<td><span class="dot" style="display:inline-block;background:rgb{COLORS.get(label, (200, 200, 200))}"></span> '
                       f'<span class="ckey">{label}</span></td><td class="num">{score:.3f}</td>'
                       f'<td class="num">{w:.0f}×{h:.0f}</td><td>{"เก็บ" if keep else "ตัด"}</td><td>{res}</td></tr>')
        missed = [a for a, h in zip(r["boxes"], a_hit) if not h]
        miss_txt = "".join(f'<div class="miss">? พลาด {a["label"]}</div>' for a in missed)
        if not r["boxes"]:
            miss_txt = '<div class="muted">ภาพนี้ปกติ ไม่มีตำหนิ — ทุกกรอบที่เก็บคือทักผิด</div>'
        table = ('<table class="dbg"><tr><th>#</th><th>class</th><th>คะแนน</th><th>ขนาด</th><th>my_filter</th><th>ผล</th></tr>'
                 + "".join(trs) + "</table>" if trs else '<div class="muted">ไม่มีกรอบเลย</div>')
        self.gal.value = (f'<div class="one"><img src="data:image/jpeg;base64,{_jpeg_b64(out, 720)}">'
                          f'<div>{table}{miss_txt}</div></div>')
        self.foot.value = ('<div class="k">ตัวเลขบนภาพตรงกับแถวในตาราง · กรอบเทา = my_filter ตัดทิ้ง · '
                           '“ตัดตัวถูกทิ้ง” = ลองลดเกณฑ์ของแบบนั้น · ไล่ภาพด้วย ◀ ▶</div>')
