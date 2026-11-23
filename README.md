# SIITec Hackathon 2569 — Vision-Language AI for Smart Factory

รอบคัดเลือกออนไลน์ (28 พ.ย. 2569): ใช้ OWLv2 หาตำหนิของชิ้นงานด้วยการ **พิมพ์คำค้น** แล้วเขียน Python (`my_filter`, `my_draw`) ต่อเอง

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIITec-Hackathon-2026/round1/blob/main/round1.ipynb)

เปิด `round1.ipynb` ใน Colab → กด ▶ ที่เซลล์เดียวของโน้ตบุ๊ก (โน้ตบุ๊กขอ T4 GPU ให้เอง)

### อุ่นเครื่องก่อน: ลองเล่น OWLv2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIITec-Hackathon-2026/round1/blob/main/playground.ipynb)

`playground.ipynb` — พิมพ์คำ (a person, a dog, a car …) แล้วดูว่า AI หาเจอตรงไหน ในรูปตัวอย่างหรือรูปที่อัปโหลดเอง
รูปตัวอย่างใน `assets/playground/` มาจาก Wikimedia Commons (CC0 / public domain) ดูเครดิตใน `assets/playground/CREDITS.txt`

## ข้อมูล

| ไฟล์ | มีอะไร |
|---|---|
| `train.zip` ([Releases](https://github.com/SIITec-Hackathon-2026/round1/releases)) | ภาพสำหรับฝึก 217 ภาพ (มีภาพชิ้นงานปกติปนอยู่) + เฉลย `labels.json` — notebook โหลดให้เอง |
| ชุดทดสอบลับ | ไม่แจก — กรรมการรันไฟล์ส่งงานของแต่ละทีมกับชุดนี้ และเปิดเผยหลังประกาศผล |

หน้าจอเป็นแบบ editor: ซ้ายมีแท็บ **วิธีเล่น** (อ่านก่อน) กับสองไฟล์ `prompts.txt` (คำค้น) และ `script.py` (`my_filter` มีตัวอย่างให้แก้ต่อ, `my_draw`) · ขวาคือภาพ · ล่างคือผลและข้อผิดพลาด · กด **Run** หรือ Ctrl+Enter

ปุ่ม **บันทึก** ดาวน์โหลดไฟล์ `siitec_round1_<ทีม>.json` (คำค้น + Script) ไว้แนบตอนส่งงานตามช่องทางที่ผู้จัดแจ้ง ปุ่ม **เปิด** ใช้ไฟล์นี้ทำงานต่อ

`labels.json`:

```json
{"classes": ["bottle_broken_large", "..."],
 "images": [{"file": "train_0001.jpg", "width": 900, "height": 900, "category": "bottle",
             "boxes": [{"label": "bottle_broken_large", "box": [x1, y1, x2, y2]}]}]}
```

ภาพที่ `boxes` ว่าง คือชิ้นงานปกติ ไม่มีตำหนิ

## ที่มาและสัญญาอนุญาตของข้อมูล

ภาพทั้งหมดมาจาก **MVTec AD** (หมวด screw, metal_nut, bottle) © MVTec Software GmbH
ใช้ภายใต้ [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (ข้อความเต็มใน `LICENSE-DATA.txt` ในแต่ละ zip) — ห้ามใช้เชิงพาณิชย์

สิ่งที่เราเปลี่ยนจากต้นฉบับ: แบ่งชุดฝึก/ทดสอบใหม่, แปลง mask เป็นกรอบ (รวมจุดที่อยู่ใกล้กันเป็นกรอบเดียว),
ตั้งชื่อไฟล์ใหม่, บันทึกเป็น JPEG และหมุน/กลับด้านภาพชุดทดสอบ ข้อมูลที่ดัดแปลงนี้แจกภายใต้ CC BY-NC-SA 4.0 เช่นเดียวกัน

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IJCV 129(4):1038-1059, 2021.
>
> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019, 9584-9592.
