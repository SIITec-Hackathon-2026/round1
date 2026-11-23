# ตัดกรอบที่ชิดขอบภาพ: จุดเสียมักอยู่บนชิ้นงาน
def my_filter(image, detections):
    h, w = image.shape[:2]
    m = 0.02 * max(h, w)         # ระยะจากขอบ
    keep = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        if d["score"] < THRESHOLD[d["label"]]:
            continue
        edge = x1 < m or y1 < m or x2 > w - m or y2 > h - m
        if edge and d["label"] != "metal_nut_flip":
            continue             # น็อตคว่ำ กรอบคลุมทั้งชิ้น
        keep.append(d)
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep[:MAX_BOXES]
