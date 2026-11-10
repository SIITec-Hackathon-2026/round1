# SIITec Hackathon 2569 — Vision-Language AI for Smart Factory

รอบคัดเลือกออนไลน์ (28 พ.ย. 2569): ใช้ OWLv2 หาตำหนิของชิ้นงานด้วยการ **พิมพ์คำค้น** แล้วเขียน Python กรองผลต่อเอง

## ข้อมูล

| โฟลเดอร์ | มีอะไร |
|---|---|
| `data/train/images/` | ภาพสำหรับฝึก 217 ภาพ (มีภาพชิ้นงานปกติปนอยู่) |
| `data/train/labels.json` | เฉลยกรอบตำหนิของภาพฝึก |

ชุดภาพทดสอบ (ไม่มีเฉลย) ปล่อยใน Releases เวลา 10.30 น. วันแข่ง

`labels.json`:

```json
{"classes": ["bottle_broken_large", "..."],
 "images": [{"file": "train_0001.jpg", "width": 900, "height": 900, "category": "bottle",
             "boxes": [{"label": "bottle_broken_large", "box": [x1, y1, x2, y2]}]}]}
```

ภาพที่ `boxes` ว่าง คือชิ้นงานปกติ ไม่มีตำหนิ

## ที่มาและสัญญาอนุญาตของข้อมูล

ภาพทั้งหมดมาจาก **MVTec AD** (หมวด screw, metal_nut, bottle) © MVTec Software GmbH
ใช้ภายใต้ [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (ข้อความเต็มใน `data/LICENSE-DATA.txt`) — ห้ามใช้เชิงพาณิชย์

สิ่งที่เราเปลี่ยนจากต้นฉบับ: แบ่งชุดฝึก/ทดสอบใหม่, แปลง mask เป็นกรอบ (รวมจุดที่อยู่ใกล้กันเป็นกรอบเดียว),
ตั้งชื่อไฟล์ใหม่, บันทึกเป็น JPEG และหมุน/กลับด้านภาพชุดทดสอบ ข้อมูลที่ดัดแปลงนี้แจกภายใต้ CC BY-NC-SA 4.0 เช่นเดียวกัน

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IJCV 129(4):1038-1059, 2021.
>
> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019, 9584-9592.
