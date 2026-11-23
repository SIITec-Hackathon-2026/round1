# ดูสีในกรอบ: ข้างในเรียบเกินไป น่าจะเป็นพื้นหลัง
def my_filter(image, detections):
    keep = []
    for d in detections:
        if d["score"] < THRESHOLD[d["label"]]:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0 or crop.std() < 8:
            continue             # สีในกรอบแทบไม่ต่างกัน
        keep.append(d)
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep[:MAX_BOXES]
