import cv2
import numpy as np

# Tasvirni o'qish (sizning tasviringiz nomi bilan almashtiring)
# Brayl tasviri bo'lsa, uni kulrang rejimda o'qish afzalroq
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Xato: Tasvir topilmadi yoki o'qib bo'lmadi.")
else:
    # Tasvirning o'lchamlari
    height, width = img.shape
    print(f"Tasvir o'lchamlari: Kenglik={width}, Balandlik={height}")

    # Yuqori chap burchakdagi piksel qiymati (0,0 koordinata)
    pixel_0_0 = img[0, 0]
    print(f"Yuqori chap burchakdagi piksel qiymati: {pixel_0_0}")

    # O'rtadagi piksel qiymati (taxminiy)
    center_x, center_y = width // 2, height // 2
    pixel_center = img[center_y, center_x]
    print(f"Markazdagi piksel qiymati ({center_y},{center_x}): {pixel_center}")

    # Tasvirning bir qismini ko'rish (masalan, 10x10 blok)
    # Bu matritsa shaklida bo'ladi
    top_left_block = img[0:10, 0:10]
    print("\nYuqori chap burchakdagi 10x10 piksel bloki (matritsa ko'rinishida):\n", top_left_block)

    # Agar rangli tasvir bo'lganda edi (masalan, 'color_image.jpg')
    # color_img = cv2.imread('color_image.jpg', cv2.IMREAD_COLOR)
    # if color_img is not None:
    #     # Piksel qiymati (B, G, R ketma-ketligida OpenCV da)
    #     color_pixel = color_img[0, 0]
    #     print(f"\nRangli tasvirning yuqori chap burchakdagi piksel qiymati (B, G, R): {color_pixel}")