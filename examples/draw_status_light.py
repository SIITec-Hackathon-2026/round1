# ไฟสถานะ OK / NG: เขียวถ้าไม่เจอตำหนิ แดงถ้าเจอ
def my_draw(image, found, answers):
    img = image.copy()
    h, w = img.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    r = max(20, w // 14)                 # ขนาดไฟ
    c = (w - r - 15, r + 15)             # มุมขวาบน
    ok = len(found) == 0
    color = (40, 200, 90) if ok else (230, 40, 40)   # (R, G, B)
    cv2.circle(img, c, r + 4, (255, 255, 255), -1)
    cv2.circle(img, c, r, color, -1)
    text = "OK" if ok else "NG"
    s = r / 28
    (tw, th), _ = cv2.getTextSize(text, F, s, 3)
    cv2.putText(img, text, (c[0] - tw // 2, c[1] + th // 2),
                F, s, (255, 255, 255), 3, cv2.LINE_AA)
    for d in found:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img
