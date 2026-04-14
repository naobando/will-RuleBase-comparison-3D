"""部品の位置を前景検出で測定"""
import cv2
import numpy as np

for label, path in [
    ("Master", "20260407/マスター画像/平面サンプル_マスター.png"),
    ("Normal", "20260407/正常/平面サンプル正常確定.bmp"),
    ("Abnormal", "20260407/異常/平面サンプル異常確定.bmp"),
]:
    img = cv2.imread(path)
    if img is None:
        print(f"{label}: NOT FOUND")
        continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(main_c)
        print(f"{label}: size={w}x{h}, part_bbox=({x},{y},{cw},{ch}), part_bottom={y+ch}, center=({x+cw//2},{y+ch//2})")
    else:
        print(f"{label}: size={w}x{h}, no contours found")
