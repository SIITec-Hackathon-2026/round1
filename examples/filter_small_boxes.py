# กรอบเล็กต้องมั่นใจมากขึ้น: กรอบจิ๋วใช้เกณฑ์สองเท่า
def my_filter(image, detections):
    h, w = image.shape[:2]
    keep = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        size = (x2 - x1) * (y2 - y1) / (w * h)
        need = THRESHOLD[d["label"]]
        if size < 0.01:          # กรอบเล็กกว่า 1% ของภาพ
            need = need * 2      # ต้องมั่นใจเป็นสองเท่า
        if d["score"] >= need:
            keep.append(d)
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep[:MAX_BOXES]
