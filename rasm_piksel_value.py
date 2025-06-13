import cv2
import numpy as np

# Kulrang tasvir yaratish (masalan, 100x100 o'lchamli qora tasvir)
img = np.zeros((100, 100), dtype=np.uint8) # Barcha piksellar 0 (qora)

# Markazdagi pikselning qiymatini o'zgartirish (masalan, oq qilish)
# Koordinatalar: y (qator), x (ustun)
center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
img[center_y, center_x] = 255 # Markazdagi pikselni oq rangga o'rnatish

# Yuqori chap burchakdagi piksel qiymatini olish
pixel_value = img[0, 0]
print(f"(0,0) koordinatadagi piksel qiymati: {pixel_value}")

# Tasvirni ko'rsatish
cv2.imshow('Ozgargan Tasvir', img)
cv2.waitKey(0)
cv2.destroyAllWindows()