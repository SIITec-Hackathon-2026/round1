# ป้ายชื่อมีพื้นหลัง: อ่านง่ายขึ้น (OpenCV เขียนได้แต่ภาษาอังกฤษ)
def my_draw(image, found, answers):
    img = image.copy()
    t = max(2, img.shape[0] // 300)
    F = cv2.FONT_HERSHEY_SIMPLEX
    for d in found:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 140, 0), t)
        text = d["label"].split("_")[-1] + f' {d["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(text, F, t / 3, 2)
        y = max(th + 8, y1)
        cv2.rectangle(img, (x1, y - th - 8), (x1 + tw + 8, y),
                      (255, 140, 0), -1)          # พื้นป้าย
        cv2.putText(img, text, (x1 + 4, y - 5), F, t / 3,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return img
