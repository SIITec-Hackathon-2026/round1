# แถบสรุปด้านล่าง: นับจุดเสีย เหมือนจอเครื่องตรวจ
def my_draw(image, found, answers):
    img = image.copy()
    h, w = img.shape[:2]
    bar = max(40, h // 12)
    cv2.rectangle(img, (0, h - bar), (w, h), (20, 20, 20), -1)
    names = sorted({d["label"].split("_")[-1] for d in found})
    text = f"defects: {len(found)}  " + " ".join(names)
    cv2.putText(img, text, (12, h - bar // 3), cv2.FONT_HERSHEY_SIMPLEX,
                bar / 45, (255, 220, 50), 2, cv2.LINE_AA)
    for d in found:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 220, 50), 3)
    return img
