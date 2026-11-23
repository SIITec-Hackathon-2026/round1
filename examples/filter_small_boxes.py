# กรอบเล็กต้องมั่นใจมากขึ้น: my_filter ตั้งต้น + กรอบจิ๋วใช้เกณฑ์สองเท่า (บรรทัด  # NEW
# วางทับ def my_filter เดิมทั้งก้อน (ใช้ THRESHOLD, MIN_SIZE, MAX_SIZE, OVERLAP, MAX_BOXES, iou ที่มีอยู่แล้วใน script.py)
def my_filter(image, detections):
    h, w = image.shape[:2]
    keep = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        size = (x2 - x1) * (y2 - y1) / (w * h)
        need = THRESHOLD[d["label"]]
        if size < 0.01:  # NEW
            need = need * 2  # NEW
        if d["score"] < need:  # NEW
            continue
        if size < MIN_SIZE:
            continue
        if size > MAX_SIZE and d["label"] != "metal_nut_flip":
            continue
        keep.append(d)
    keep.sort(key=lambda d: d["score"], reverse=True)
    final = []
    for d in keep:
        if any(iou(d["box"], k["box"]) > OVERLAP for k in final):
            continue
        final.append(d)
    return final[:MAX_BOXES]
