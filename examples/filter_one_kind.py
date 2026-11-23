# ภาพละแบบเดียว: เก็บเฉพาะแบบที่คะแนนสูงสุดของภาพนั้น
def my_filter(image, detections):
    keep = [d for d in detections
            if d["score"] >= THRESHOLD[d["label"]]]
    if not keep:
        return []
    best = max(keep, key=lambda d: d["score"])["label"]
    keep = [d for d in keep if d["label"] == best]
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep[:MAX_BOXES]
