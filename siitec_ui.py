"""SIITec Hackathon 2569, round 1: the notebook's screen (ipywidgets), laid out like a code editor.

App() gets everything ready (GPU, data, model, picture features) and app.show()
draws the screen: the team's two files (prompts.txt, script.py) on the left, the
pictures on the right, the output panel and a status bar below. ▶ Run (or Ctrl+Enter
in a file) runs both files again.
"""
import base64
import contextlib
import html
import io
import json
import re
import signal
import threading
import time
import traceback
import unicodedata
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
EDITOR_H = 560   # height of a file in the editor, px
SCRIPT_SECONDS = 15     # running script.py itself (imports, constants)
FILTER_SECONDS = 120    # my_filter over every picture
DRAW_SECONDS = 20       # my_draw over one page of pictures
MAX_PROMPT_TOKENS = 16  # OWLv2 reads no more of a prompt

# Flat editor look: Dark Modern / Light Modern colours, no gradients, square corners
STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.siitec{--bg:#ffffff;--side:#f8f8f8;--bar:#f8f8f8;--tab:#f8f8f8;--line:#e5e5e5;--ink:#3b3b3b;--muted:#767676;
  --field:#ffffff;--fieldline:#cecece;--accent:#005fb8;--on-accent:#ffffff;--hover:#e8e8e8;--sel:#add6ff;
  --status:#005fb8;--status-bad:#c72e0f;--good:#388a34;--bad:#c72e0f;--warn:#8a6a00;--gutter:#6e7681}
@media (prefers-color-scheme: dark){.siitec{--bg:#1f1f1f;--side:#181818;--bar:#181818;--tab:#181818;--line:#2b2b2b;--ink:#cccccc;
  --muted:#9d9d9d;--field:#313131;--fieldline:#3c3c3c;--accent:#0078d4;--on-accent:#ffffff;--hover:#2a2d2e;--sel:#264f78;
  --status:#0078d4;--status-bad:#c72e0f;--good:#89d185;--bad:#f48771;--warn:#cca700;--gutter:#6e7681}}
.siitec{color:var(--ink);background:var(--bg);border:1px solid var(--line);box-sizing:border-box;width:100%;
  font-family:'Sarabun',system-ui,sans-serif;font-size:14px;line-height:1.45;
  --jp-widgets-color:var(--ink);--jp-widgets-label-color:var(--ink);--jp-widgets-input-color:var(--ink);
  --jp-widgets-input-background-color:var(--field);--jp-widgets-input-border-color:var(--fieldline);
  --jp-widgets-input-focus-border-color:var(--accent);--jp-widgets-font-size:14px;--jp-layout-color1:var(--bg);--jp-border-color1:var(--line)}
.siitec *{font-family:inherit;box-sizing:border-box}
.siitec .widget-label,.siitec label{color:var(--ink)!important}
.siitec input[type=text],.siitec select{background:var(--field)!important;color:var(--ink)!important;
  border:1px solid var(--fieldline)!important;border-radius:2px!important;font-size:13px!important;height:26px!important}
.siitec input:focus,.siitec select:focus,.siitec textarea:focus{outline:1px solid var(--accent)!important;outline-offset:-1px;border-color:var(--accent)!important}
.siitec button,.siitec .jupyter-button{border-radius:2px!important;font-weight:400!important;border:1px solid var(--fieldline)!important;
  background:transparent!important;color:var(--ink)!important;height:26px!important;line-height:24px!important;padding:0 10px!important;
  box-shadow:none!important;font-size:13px!important}
.siitec button:hover{background:var(--hover)!important}
.siitec button.primary{background:var(--accent)!important;color:var(--on-accent)!important;border-color:var(--accent)!important}
.siitec button.primary:hover{filter:brightness(1.1)}
/* title bar */
.siitec .titlebar{background:var(--bar);border-bottom:1px solid var(--line);padding:6px 10px}
.siitec .hdr{display:flex;align-items:center;gap:8px}
.siitec .logo{height:24px;display:flex;align-items:center;font-weight:700;font-size:11px;color:var(--muted);border:1px solid var(--line);padding:0 6px}
.siitec .logo img{height:24px}
.siitec .t1{font-weight:600} .siitec .t2{color:var(--muted)}
.siitec .score{font-family:'JetBrains Mono','Sarabun',monospace;font-size:20px;font-weight:600;color:var(--ink);line-height:1}
.siitec .k{font-size:12px;color:var(--muted)}
/* panes */
.siitec .pane{min-width:0;overflow:hidden}
.siitec .side{background:var(--side);border-left:1px solid var(--line)}
.siitec .tabs{background:var(--tab);border-bottom:1px solid var(--line);height:35px;flex:none!important;align-items:stretch!important}
.siitec .tabs button{height:35px!important;line-height:33px!important;border:none!important;border-right:1px solid var(--line)!important;
  border-top:1px solid transparent!important;padding:0 14px!important;color:var(--muted)!important;font-family:'JetBrains Mono','Sarabun',monospace!important;font-size:12.5px!important}
.siitec .tabs button.on{background:var(--bg)!important;color:var(--ink)!important;border-top-color:var(--accent)!important}
.siitec .tabs .grow{flex:1}
.siitec .tabs .tool{display:flex;align-items:center;gap:6px;padding:0 8px}
.siitec .siitec-code{width:100%;flex:none!important}
.siitec .siitec-code textarea{font-family:'JetBrains Mono','Sarabun',ui-monospace,monospace!important;font-size:13px!important;line-height:19px!important;
  background:var(--bg)!important;color:var(--ink)!important;border:none!important;border-radius:0!important;padding:6px 10px!important;
  tab-size:4;white-space:pre;overflow:auto;resize:none;height:100%!important}
.siitec .siitec-code textarea::selection{background:var(--sel)}
.siitec .gutter{font-family:'JetBrains Mono','Sarabun',monospace;font-size:13px;line-height:19px;color:var(--gutter);text-align:right;
  padding:6px 8px 0 10px;white-space:pre;overflow:hidden;user-select:none;min-width:44px;flex:none;background:var(--bg)}
.siitec .viewbar{padding:4px 8px;border-bottom:1px solid var(--line);height:35px;flex:none!important;align-items:center}
.siitec .view{overflow:auto;padding:8px}
.siitec .seg button{border-radius:0!important;margin:0!important}
.siitec .seg button+button{border-left:none!important}
.siitec .seg button.on{background:var(--accent)!important;color:var(--on-accent)!important;border-color:var(--accent)!important}
/* output panel and status bar */
.siitec .panel{border-top:1px solid var(--line);padding:4px 12px 8px}
.siitec .ptitle{font-size:11px;letter-spacing:.6px;color:var(--muted);text-transform:uppercase;padding:2px 0 4px}
.siitec .f1grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));column-gap:22px;font-family:'JetBrains Mono','Sarabun',monospace;font-size:12.5px}
.siitec .f1grid div{display:flex;gap:8px;align-items:center;white-space:nowrap}
.siitec .f1grid .c{overflow:hidden;text-overflow:ellipsis;flex:1}
.siitec .statusbar{background:var(--status);color:#fff;font-size:12px;padding:0 10px;height:22px;display:flex;align-items:center;gap:16px;white-space:nowrap;overflow:hidden}
.siitec .statusbar.bad{background:var(--status-bad)}
.siitec .statusbar .r{margin-left:auto}
/* text marks */
.siitec .up{color:var(--good)} .siitec .down{color:var(--bad)} .siitec .muted{color:var(--muted)}
.siitec .ok{color:var(--good)} .siitec .miss{color:var(--bad)} .siitec .fa{color:var(--warn)}
.siitec .err{color:var(--bad)}
.siitec .noteline{font-family:'JetBrains Mono','Sarabun',monospace;font-size:12.5px;color:var(--warn);white-space:pre-wrap}
.siitec .printed{font-family:'JetBrains Mono','Sarabun',monospace;font-size:12.5px;white-space:pre-wrap;max-height:160px;overflow:auto;
  border-left:2px solid var(--line);padding-left:8px;margin-bottom:6px}
.siitec .titlebar > *,.siitec .viewbar > *{flex-shrink:0}
.siitec .titlebar .hdr{white-space:nowrap}
.siitec .viewbar .widget-html{white-space:nowrap}
.siitec .errline{font-family:'JetBrains Mono','Sarabun',monospace;font-size:12.5px;color:var(--bad);white-space:pre-wrap}
.siitec .dot{width:9px;height:9px;border-radius:50%;flex:none;display:inline-block}
.siitec .ckey{font-family:'JetBrains Mono','Sarabun',monospace;font-size:12px}
.siitec code{font-family:'JetBrains Mono','Sarabun',monospace;font-size:12.5px}
/* pictures */
.siitec .gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:6px}
.siitec .shot{border:1px solid var(--line);background:var(--bg);cursor:pointer}
.siitec .shot:hover{outline:1px solid var(--accent)}
.siitec .shot img{width:100%;display:block;aspect-ratio:1/1;object-fit:cover;background:#000}
.siitec .shot .cap{display:flex;gap:6px;align-items:center;padding:2px 6px;font-size:11.5px;font-family:'JetBrains Mono','Sarabun',monospace}
.siitec .shot .fn{color:var(--muted);margin-right:auto}
.siitec .one img{width:100%;display:block;margin-bottom:8px}
.siitec table.dbg{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:6px}
.siitec table.dbg th{color:var(--muted);font-weight:400;text-align:left;padding:3px 6px;border-bottom:1px solid var(--line);font-size:11.5px}
.siitec table.dbg td{padding:3px 6px;border-bottom:1px solid var(--line)}
.siitec .num{font-family:'JetBrains Mono','Sarabun',monospace;text-align:right;white-space:nowrap}
.siitec .prog{height:2px;background:var(--line);overflow:hidden;max-width:360px;margin:4px 0 0 26px}
.siitec .prog i{display:block;height:100%;background:var(--accent)}
.siitec .setup{padding:10px 14px}
.siitec .jump{display:none!important}
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

PROMPTS_HEAD = """# คำค้นภาษาอังกฤษ บรรทัดละแบบ ใต้ [ชื่อแบบ]
# ขึ้นต้นด้วย -  = "ไม่ใช่สิ่งนี้"    ·    หลัง # = หมายเหตุ ไม่ถูกใช้
# เกณฑ์ความมั่นใจตั้งใน script.py
"""


def prompts_to_text(prompts, classes):
    lines = [PROMPTS_HEAD]
    for c in classes:
        lines.append(f"[{c}]".ljust(24) + f"  # {so.PART_TH[so.CLASS_INFO[c][0]]}: {so.CLASS_INFO[c][1]}")
        lines += prompts.get(c, [])
        lines.append("")
    return "\n".join(lines)


def parse_prompts(text, classes):
    """prompts.txt -> ({class: [prompt, ...]}, [(line, message), ...])."""
    prompts, errors, cur = {c: [] for c in classes}, [], None
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"\[\s*([A-Za-z0-9_]+)\s*\]", line)
        if m:
            cur = m.group(1)
            if cur not in prompts:
                errors.append((n, f"ไม่มีแบบชื่อ {cur}"))
                cur = None
            continue
        if line.startswith("["):
            errors.append((n, "หัวข้อต้องเป็น [ชื่อแบบ]"))
        elif cur is None:
            errors.append((n, "คำค้นนี้ไม่อยู่ใต้ [ชื่อแบบ] ใด"))
        else:
            prompts[cur].append(line)
    return prompts, errors


# The editor's keys and line numbers: Tab types four spaces, Ctrl+Enter presses ▶ Run,
# a picture clicked in the grid opens on its own.
EDITOR_JS = """
<script>
(function(){
  if (window.__siitecEditor) return; window.__siitecEditor = 1;
  function nums(t, g){
    var n = t.value.split('\\n').length;
    if (g.dataset.n != n){ var s = []; for (var i = 1; i <= n; i++) s.push(i); g.textContent = s.join('\\n'); g.dataset.n = n; }
    g.scrollTop = t.scrollTop;
  }
  function hook(){
    document.querySelectorAll('.siitec-code textarea:not([data-ed])').forEach(function(t){
      t.dataset.ed = 1; t.spellcheck = false;
      var g = document.createElement('div'); g.className = 'gutter';
      t.parentElement.insertBefore(g, t);
      var up = function(){ nums(t, g); };
      t.addEventListener('input', up); t.addEventListener('scroll', up); up();
      setInterval(up, 400);                      // the value can also change from Python
      t.addEventListener('keydown', function(e){
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)){
          e.preventDefault();
          t.dispatchEvent(new Event('input', {bubbles: true}));
          var b = t.closest('.siitec').querySelector('button.run-btn');
          if (b) setTimeout(function(){ b.click(); }, 60);
          return;
        }
        if (e.key !== 'Tab') return;
        e.preventDefault();
        var s = t.selectionStart, v = t.value;
        t.value = v.slice(0, s) + '    ' + v.slice(t.selectionEnd);
        t.selectionStart = t.selectionEnd = s + 4;
        t.dispatchEvent(new Event('input', {bubbles: true}));
      });
    });
  }
  document.addEventListener('click', function(e){
    var s = e.target.closest && e.target.closest('.siitec .shot[data-i]');
    if (!s) return;
    var inp = s.closest('.siitec').querySelector('.jump input');
    if (!inp) return;
    inp.value = s.dataset.i + ':' + Date.now();
    inp.dispatchEvent(new Event('input', {bubbles: true}));
  });
  new MutationObserver(hook).observe(document.body, {childList: true, subtree: true});
  hook();
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


class _Printed(io.StringIO):
    """What the team's code prints, kept for the output panel."""

    def lines(self, n=30):
        out = self.getvalue().rstrip("\n").split("\n") if self.getvalue().strip() else []
        return (["…"] if len(out) > n else []) + out[-n:]


@contextlib.contextmanager
def _team_code(printed, seconds):
    """Run the team's code: catch its prints, stop it after `seconds` (an endless loop must
    not freeze the screen). The time limit needs the main thread and SIGALRM (Colab has both)."""
    timed = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
    if timed:
        def stop(*_):
            raise TimeoutError(f"ทำงานนานเกิน {seconds} วินาที — มีลูปไม่จบหรือเปล่า")
        old = signal.signal(signal.SIGALRM, stop)
        signal.alarm(seconds)
    try:
        with contextlib.redirect_stdout(printed):
            yield
    finally:
        if timed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _team_error(e):
    """An error of the team's code as text; Ctrl+C / the Stop button counts as one too."""
    if isinstance(e, KeyboardInterrupt):
        return "หยุดกลางคัน (กด Stop)"
    return _err_text(e)


def _root(children):
    box = W.VBox([_html(STYLE)] + children)
    box.add_class("siitec")
    return box


def default_draw(image, found, answers):
    """Built-in drawing, used when the script has no my_draw (or it fails)."""
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
        box.add_class("setup")
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
        prompts = saved.get("prompts") or {c: [self._default_prompt(c)] for c in self.classes}
        self.prompts_text = saved.get("prompts_text") or prompts_to_text(prompts, self.classes)
        self.prompts, self.prompt_errors = parse_prompts(self.prompts_text, self.classes)
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
                                                  "prompts_text": self.prompts_text, "script": self.script},
                                                 ensure_ascii=False, indent=1))
        except Exception:
            pass

    def _read_inputs(self):
        self.prompts_text = self.prompt_area.value
        self.prompts, self.prompt_errors = parse_prompts(self.prompts_text, self.classes)
        self.long_prompts = [p for ps in self.prompts.values() for p in ps if self._tokens(p) > MAX_PROMPT_TOKENS]
        self.script = self.script_area.value
        self.fns, self.script_error = {}, None
        self.printed = _Printed()
        try:
            ns = {"np": np, "cv2": cv2}
            with _team_code(self.printed, SCRIPT_SECONDS):
                exec(compile(self.script, "<script>", "exec"), ns)
            self.fns = {k: ns[k] for k in ("my_filter", "my_draw") if callable(ns.get(k))}
        except (Exception, KeyboardInterrupt) as e:
            self.script_error = _team_error(e)

    def _tokens(self, prompt):
        """Tokens OWLv2 sees in a prompt, start and end marks included."""
        try:
            return len(self.finder.proc.tokenizer(prompt.lstrip("-").strip())["input_ids"])
        except Exception:
            return 0

    # ------------------------------------------------------------ running
    def _run(self, ds):
        self._read_inputs()
        ds.candidates = self.finder.find(ds.cache, ds.records, self.prompts)
        # without a (working) my_filter every class keeps boxes scoring START_THRESHOLD or more
        found = so.apply_thresholds(ds.candidates, {c: START_THRESHOLD for c in self.classes})
        self.filter_error = None
        fn = self.fns.get("my_filter")
        if fn and not self.script_error:
            try:
                with _team_code(self.printed, FILTER_SECONDS):
                    found = so.apply_filter(fn, ds.dir, ds.records, ds.candidates)
            except so.FilterOutputError as e:
                self.filter_error = str(e)
            except (Exception, KeyboardInterrupt) as e:
                self.filter_error = _team_error(e)
        ds.found = found
        ds.score, ds.per, ds.marks = so.evaluate(ds.records, found, ds.classes)

    def _press(self):
        t = time.time()
        self.statusbar.value = '<div class="statusbar">⏳ กำลังรัน…</div>'
        try:
            prev = self.train.score
            self._run(self.train)
            if self.test is not None:
                self._run(self.test)
            self._autosave()
            self._show_result(prev, time.time() - t)
            self.gallery.refresh()
        except (Exception, KeyboardInterrupt) as e:
            self.statusbar.value = f'<div class="statusbar bad">✕ {html.escape(_team_error(e))}</div>'

    def _show_result(self, prev, secs):
        s = self.train.score
        self.best = s if self.best is None else max(self.best, s)
        delta = ""
        if prev is not None and abs(s - prev) > 1e-6:
            delta = f'<span class="{"up" if s > prev else "down"}">{"▲" if s > prev else "▼"}{abs(s - prev):.3f}</span>'
        self.score_html.value = (f'<div style="text-align:right"><div class="score">{s:.3f}</div>'
                                 f'<div class="k">{delta} ชุดฝึก · ดีสุด {self.best:.3f}</div></div>')

        problems = [f"prompts.txt บรรทัด {n}: {m}" for n, m in self.prompt_errors]
        if self.script_error:
            problems.append(f"script.py  {self.script_error}  → ใช้เกณฑ์ {START_THRESHOLD} แทน")
        elif self.filter_error:
            problems.append(f"my_filter  {self.filter_error}  → ใช้เกณฑ์ {START_THRESHOLD} แทน")
        rows = []
        for c in self.classes:
            v = self.train.per[c]
            rows.append(f'<div><span class="dot" style="background:rgb{COLORS[c]}"></span><span class="c">{c}</span>'
                        f'<b>{v["f1"]:.2f}</b><span class="muted">✓{v["tp"]} ✗{v["fp"]} ?{v["fn"]}</span></div>')
        notes = []
        if not self.script_error and "my_filter" not in self.fns:
            notes.append(f"script.py ไม่มี def my_filter — ใช้เกณฑ์ {START_THRESHOLD} ทุกแบบ")
        if not self.script_error and "my_draw" not in self.fns:
            notes.append("script.py ไม่มี def my_draw — วาดภาพแบบตั้งต้น")
        notes += [f"prompts.txt คำค้นยาวเกิน {MAX_PROMPT_TOKENS - 2} คำ ส่วนท้ายถูกตัด: {p}" for p in self.long_prompts]
        err = "".join(f'<div class="errline">✕ {html.escape(p)}</div>' for p in problems)
        err += "".join(f'<div class="noteline">! {html.escape(n)}</div>' for n in notes)
        printed = self.printed.lines()
        if printed:
            err += ('<div class="ptitle" style="padding-top:6px">print</div><div class="printed">'
                    + html.escape("\n".join(printed)) + "</div>")
        self.output.value = (f'<div class="ptitle">Output · F1 ของแต่ละแบบ (ชุดฝึก) · ✓ ถูก ✗ ทักผิด ? พลาด</div>'
                             f'{err}<div class="f1grid">{"".join(rows)}</div>')

        flt = "my_filter ✓" if self.fns.get("my_filter") and not self.filter_error else f"เกณฑ์ {START_THRESHOLD}"
        test = f" · ชุดทดสอบ {self.test.score:.3f}" if self.test is not None else ""
        state = f"✕ {len(problems)} ปัญหา" if problems else "✓ ไม่มีปัญหา"
        self.statusbar.value = (f'<div class="statusbar{" bad" if problems else ""}"><span>{state}</span>'
                                f'<span>{flt}</span><span>{secs:.1f} วินาที</span>'
                                f'<span class="r">ชุดฝึก {s:.3f}{test}</span></div>')

    # ------------------------------------------------------------ screen
    def show(self):
        run = _button("▶ Run", primary=True)
        run.add_class("run-btn")
        run.tooltip = "Ctrl+Enter"
        run.on_click(lambda _: self._press())
        self.score_html = _html('<div class="score">–</div>')
        self.team_box = W.Text(value=self.team, placeholder="ชื่อทีม", layout=W.Layout(width="160px"))
        save, load, self.file_msg = self._file_tools()
        self.save_button, self.load_button = save, load
        title = W.HBox([_html(HEADER), W.HBox(layout=W.Layout(flex="1")), self.team_box, save, load,
                        self.score_html, run],
                       layout=W.Layout(align_items="center", grid_gap="8px"))
        title.add_class("titlebar")

        self.prompt_area = self._code(self.prompts_text)
        self.script_area = self._code(self.script)
        files = {"prompts.txt": self.prompt_area, "script.py": self.script_area}
        tab_buttons = {}

        def open_file(name):
            for n, b in tab_buttons.items():
                files[n].layout.display = "flex" if n == name else "none"
                (b.add_class if n == name else b.remove_class)("on")

        for name in files:
            b = W.Button(description=name, layout=W.Layout(width="auto"))
            b.on_click(lambda _, n=name: open_file(n))
            tab_buttons[name] = b
        reset = _button("โค้ดตั้งต้น")
        reset.tooltip = "ใส่ script.py ตั้งต้นกลับ"

        def do_reset(_):
            if reset.description != "กดอีกครั้งเพื่อยืนยัน":
                reset.description = "กดอีกครั้งเพื่อยืนยัน"
                return
            reset.description = "โค้ดตั้งต้น"
            self.script_area.value = SCRIPT_TEMPLATE
            open_file("script.py")
            self._press()

        reset.on_click(do_reset)
        tool = W.HBox([reset])
        tool.add_class("tool")
        bar = W.HBox(list(tab_buttons.values()) + [W.HBox(layout=W.Layout(flex="1")), tool])
        bar.add_class("tabs")
        editor = W.VBox([bar, self.prompt_area, self.script_area],
                        layout=W.Layout(flex="1 1 0", height=f"{EDITOR_H + 35}px"))
        editor.add_class("pane")
        open_file("prompts.txt")

        sets = [("ชุดฝึก", "train")] + ([("🔓 ชุดทดสอบ", "test")] if self.has_test else [])
        self.view_set = "train"
        self.gallery = Gallery(lambda: self.test if self.view_set == "test" else self.train,
                               lambda: self.fns.get("my_draw"), sets, self._pick_set, EDITOR_H)
        side = W.VBox([self.gallery.bar, self.gallery.box],
                      layout=W.Layout(flex="1 1 0", height=f"{EDITOR_H + 35}px"))
        side.add_class("pane")
        side.add_class("side")
        main = W.HBox([editor, side])

        self.output = _html('<div class="ptitle">Output</div>')
        out = W.VBox([self.file_msg, self.output])
        out.add_class("panel")
        self.statusbar = _html('<div class="statusbar">พร้อม</div>')

        self.setup_view.layout.display = "none"     # the ready list has done its job
        display(_root([title, main, out, self.statusbar, self.gallery.jump]))
        display(HTML(EDITOR_JS))
        self._press()

    def _pick_set(self, name):
        """The picture pane's set changed (train / hidden test)."""
        if name == "test" and self.test is None:
            self.statusbar.value = '<div class="statusbar">⏳ กำลังเตรียมชุดทดสอบ…</div>'
            try:
                self.test = Dataset("test", self.finder, self.folder)
            except Exception as e:
                self.statusbar.value = f'<div class="statusbar bad">{html.escape(str(e))}</div>'
                return
            self.view_set = name
            self._press()
            return
        self.view_set = name
        self.gallery.refresh()

    @staticmethod
    def _code(text):
        area = W.Textarea(value=text, layout=W.Layout(width="100%", height=f"{EDITOR_H}px"))
        area.add_class("siitec-code")
        return area

    def _file_tools(self):
        save = _button("บันทึก")
        save.tooltip = "ดาวน์โหลดไฟล์คำค้น + Script ของทีม"
        load = W.FileUpload(accept=".json", multiple=False, description="เปิด", icon="",
                            layout=W.Layout(width="auto"))
        msg = _html()

        def do_save(_):
            team = self.team_box.value.strip()
            if not team:
                msg.value = '<span class="err">ใส่ชื่อทีมก่อน</span>'
                return
            self._read_inputs()
            exp = so.make_export(team, self.prompts, self.script, round(self.train.score or 0, 4))
            exp["prompts_text"] = self.prompts_text
            safe = "".join(ch if ch.isalnum() or unicodedata.category(ch).startswith("M") else "_"
                           for ch in team)[:40] or "team"   # Thai vowel marks are category M
            path = Path(f"siitec_round1_{safe}.json")
            path.write_text(json.dumps(exp, ensure_ascii=False, indent=1))
            self._autosave()
            msg.value = f'<span class="ok">✓ {path.name}</span>'
            try:
                from google.colab import files
                files.download(str(path))
            except Exception:
                msg.value += f' <a href="{path.name}" download style="color:var(--accent)">ดาวน์โหลด</a>'

        def do_load(change):
            files = change["new"]
            item = files[0] if isinstance(files, (list, tuple)) else next(iter(files.values()), None)
            if not item:
                return
            try:
                load.value = ()     # so picking the same file again is seen as a change
            except Exception:
                pass
            try:
                data = json.loads(bytes(item["content"]).decode("utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("prompts", {}), dict) \
                        or not isinstance(data.get("script", ""), str):
                    raise ValueError
            except Exception:
                msg.value = '<span class="err">✕ ไฟล์นี้ไม่ใช่ไฟล์ที่ปุ่ม “บันทึก” ทำไว้</span>'
                return
            try:
                self.prompt_area.value = data.get("prompts_text") or prompts_to_text(data.get("prompts", {}), self.classes)
                if data.get("script"):
                    self.script_area.value = data["script"]
                self.team_box.value = data.get("team", self.team_box.value)
                msg.value = '<span class="ok">✓ เปิดไฟล์แล้ว</span>'
                self._press()
            except Exception:
                msg.value = '<span class="err">✕ ไฟล์นี้ไม่ใช่ไฟล์ที่ปุ่ม “บันทึก” ทำไว้ หรือถูกแก้จนเสีย</span>'

        save.on_click(do_save)
        load.observe(do_load, "value")
        return save, load, msg


class Gallery:
    """The picture pane: a page of pictures drawn with my_draw, or one picture with every candidate box."""

    def __init__(self, get_ds, get_draw, sets, on_set, height):
        self.get_ds, self.get_draw, self.page, self.one = get_ds, get_draw, 0, 0
        self.draw_problem, self.printed = None, _Printed()
        self.mode = "grid"
        self.seg = {m: W.Button(description=t, layout=W.Layout(width="auto")) for m, t in (("grid", "หลายภาพ"), ("one", "ทีละภาพ"))}
        for m, b in self.seg.items():
            b.on_click(lambda _, m=m: self._set_mode(m))
        seg = W.HBox(list(self.seg.values()))
        seg.add_class("seg")
        self.set_pick = W.Dropdown(options=sets, value="train", layout=W.Layout(width="104px",
                                   display="flex" if len(sets) > 1 else "none"))
        self.set_pick.observe(lambda ch: on_set(ch["new"]), "value")
        self.pick = W.Dropdown(options=[("ทุกภาพ", "all"), ("มีที่พลาด", "miss"), ("มีที่ทักผิด", "fa"), ("ถูกหมด", "ok"),
                                        ("ขวด", "bottle"), ("น็อต", "metal_nut"), ("สกรู", "screw")],
                               layout=W.Layout(width="100px"))
        self.prev, self.next = _button("◀"), _button("▶")
        self.where, self.gal, self.msg = _html(), _html(), _html()
        self.pick.observe(lambda _: self._go(0), "value")
        self.prev.on_click(lambda _: self._go(-1, step=True))
        self.next.on_click(lambda _: self._go(1, step=True))
        self.bar = W.HBox([seg, self.set_pick, self.pick, W.HBox(layout=W.Layout(flex="1")), self.prev, self.where, self.next],
                          layout=W.Layout(grid_gap="6px"))
        self.bar.add_class("viewbar")
        self.foot = _html()
        self.box = W.VBox([self.msg, self.gal, self.foot], layout=W.Layout(height=f"{height}px", grid_gap="6px"))
        self.box.add_class("view")
        self.jump = W.Text()          # a clicked grid picture arrives here as "<record index>:<time>"
        self.jump.add_class("jump")
        self.jump.observe(self._jumped, "value")
        self._set_mode("grid", refresh=False)

    def _set_mode(self, m, refresh=True):
        self.mode = m
        for k, b in self.seg.items():
            (b.add_class if k == m else b.remove_class)("on")
        if refresh:
            self.refresh()

    def _jumped(self, ch):
        try:
            i = int(ch["new"].split(":")[0])
        except ValueError:
            return
        ds = self.get_ds()
        if ds is None or ds.marks is None:
            return
        pick = self._picked(ds)
        if i in pick:
            self.one = pick.index(i)
            self._set_mode("one")

    def _go(self, n, step=False):
        if self.mode == "grid":
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
        """my_draw's picture; after its first failure on a page the rest use the built-in drawing."""
        draw, out = self.get_draw(), None
        if draw and not self.draw_problem:
            try:
                with _team_code(self.printed, DRAW_SECONDS):
                    res = draw(img.copy(), [dict(d, box=list(d["box"])) for d in ds.found[i]],
                               [dict(a, box=list(a["box"])) for a in ds.records[i]["boxes"]])
                if res is None:
                    self.draw_problem = "my_draw ไม่ได้ return ภาพ — ลืม return img หรือเปล่า"
                else:
                    a = np.asarray(res)
                    if a.ndim != 3 or a.shape[2] != 3 or a.shape[:2] != img.shape[:2]:
                        self.draw_problem = (f"my_draw ต้อง return ภาพสีขนาดเดิม {img.shape[0]}×{img.shape[1]}×3 "
                                             f"แต่ได้ {'×'.join(map(str, a.shape)) or type(res).__name__}")
                    else:
                        out = np.clip(a, 0, 255).astype(np.uint8)
            except (Exception, KeyboardInterrupt) as e:
                self.draw_problem = "my_draw  " + _team_error(e)
        if out is None:
            out = default_draw(img, ds.found[i], ds.records[i]["boxes"])
        return out

    def refresh(self):
        """Draw the pane again; a failure here is shown in the pane, never stops the screen."""
        self.draw_problem, self.printed = None, _Printed()
        try:
            self._refresh()
        except (Exception, KeyboardInterrupt) as e:
            self.gal.value = f'<div class="errline">✕ {html.escape(_team_error(e))}</div>'
        msg = ""
        if self.draw_problem:
            msg += f'<div class="errline">✕ {html.escape(self.draw_problem)}  → ใช้แบบตั้งต้น</div>'
        printed = self.printed.lines(10)
        if printed:
            msg += ('<div class="ptitle">print จาก my_draw</div><div class="printed">'
                    + html.escape("\n".join(printed)) + "</div>")
        self.msg.value = msg

    def _refresh(self):
        ds = self.get_ds()
        if ds is None or ds.marks is None:
            return
        pick = self._picked(ds)
        if self.mode == "one":
            return self._show_one(ds, pick)
        pages = max(1, -(-len(pick) // PER_PAGE))
        self.page = min(max(self.page, 0), pages - 1)
        self.where.value = f'<span class="k">{self.page + 1}/{pages}</span>'
        shots = []
        for i in pick[self.page * PER_PAGE:(self.page + 1) * PER_PAGE]:
            r, (f_hit, a_hit) = ds.records[i], ds.marks[i]
            out = self._drawn(ds, i, so.read_rgb(Path(ds.dir) / r["file"]))
            ok, miss, fa = f_hit.count(True), a_hit.count(False), f_hit.count(False)
            marks = (f'<span class="ok">✓{ok}</span>' if ok else "") + (f'<span class="miss">?{miss}</span>' if miss else "") + \
                    (f'<span class="fa">✗{fa}</span>' if fa else "")
            shots.append(f'<div class="shot" data-i="{i}"><img src="data:image/jpeg;base64,{_jpeg_b64(out, 360)}">'
                         f'<div class="cap"><span class="fn">{r["file"][:-4]}</span>{marks or "<span class=ok>ปกติ</span>"}</div></div>')
        self.gal.value = f'<div class="gal">{"".join(shots)}</div>' if shots else '<span class="k">ไม่มีภาพ</span>'
        self.foot.value = '<div class="k">วาดด้วย <code>my_draw</code> · คลิกภาพเพื่อดูทุกกรอบของภาพนั้น</div>'

    def _show_one(self, ds, pick):
        """One picture: my_draw's drawing with every box numbered, and a table of the candidates."""
        if not pick:
            self.where.value, self.gal.value, self.foot.value = "", '<span class="k">ไม่มีภาพ</span>', ""
            return
        self.one = min(max(self.one, 0), len(pick) - 1)
        i = pick[self.one]
        r, (f_hit, a_hit) = ds.records[i], ds.marks[i]
        self.where.value = f'<span class="k">{self.one + 1}/{len(pick)}</span>'
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
                       f'<td><span class="dot" style="background:rgb{COLORS.get(label, (200, 200, 200))}"></span> '
                       f'<span class="ckey">{label}</span></td><td class="num">{score:.3f}</td>'
                       f'<td class="num">{w:.0f}×{h:.0f}</td><td>{"เก็บ" if keep else "ตัด"}</td><td>{res}</td></tr>')
        missed = [a for a, h in zip(r["boxes"], a_hit) if not h]
        miss_txt = "".join(f'<div class="miss">? พลาด {a["label"]}</div>' for a in missed)
        if not r["boxes"]:
            miss_txt = '<div class="muted">ภาพนี้ปกติ ไม่มีตำหนิ — ทุกกรอบที่เก็บคือทักผิด</div>'
        table = ('<table class="dbg"><tr><th>#</th><th>class</th><th>คะแนน</th><th>ขนาด</th><th>my_filter</th><th>ผล</th></tr>'
                 + "".join(trs) + "</table>" if trs else '<div class="muted">ไม่มีกรอบเลย</div>')
        self.gal.value = (f'<div class="one"><div class="k">{r["file"]} · {so.PART_TH[r["category"]]}</div><img src="data:image/jpeg;base64,{_jpeg_b64(out, 720)}">'
                          f'{table}{miss_txt}</div>')
        self.foot.value = ('<div class="k">ตัวเลขบนภาพตรงกับแถวในตาราง · กรอบเทา = my_filter ตัดทิ้ง · '
                           '“ตัดตัวถูกทิ้ง” = ลองลดเกณฑ์ของแบบนั้น</div>')
