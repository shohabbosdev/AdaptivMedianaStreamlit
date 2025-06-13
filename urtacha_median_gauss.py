import cv2
import numpy as np
import matplotlib.pyplot as plt

# Brayl tasvirini yuklash (o'zingizning fayl nomingizni kiriting)
# Dastlab, Brayl tasviringizga sun'iy shovqin qo'shib olishingiz mumkin
# yoki shunchaki asl tasvir bilan ishlashingiz mumkin
try:
    img_braille_orig = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_orig is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Rasmni yuklashda xato! 'braille_sample_noisy.png' fayli topilmadi.")
    # Agar rasm topilmasa, test uchun sun'iy rasm yaratamiz
    img_braille_orig = np.zeros((100, 200), dtype=np.uint8)
    # Ba'zi "nuqtalarni" qo'shamiz
    img_braille_orig[20:30, 20:30] = 255
    img_braille_orig[50:60, 40:50] = 255
    img_braille_orig[80:90, 60:70] = 255
    # Sun'iy shovqin qo'shamiz (tuz va murch)
    row, col = img_braille_orig.shape
    num_pixels = row * col
    for i in range(num_pixels // 20): # 5% shovqin
        y_coord = np.random.randint(0, row - 1)
        x_coord = np.random.randint(0, col - 1)
        img_braille_orig[y_coord, x_coord] = 255 if np.random.rand() < 0.5 else 0
    print("Test uchun sun'iy shovqinli Brayl tasviri yaratildi.")


# 1. O'rtacha filtr
# kernel_size (3,3) yoki (5,5) bo'lishi mumkin. Katta qiymat ko'proq silliqlaydi.
mean_filtered_img = cv2.blur(img_braille_orig, (5,5))

# 2. Median filtr
# kernel_size toq son bo'lishi kerak (3, 5, 7 ...)
median_filtered_img = cv2.medianBlur(img_braille_orig, 5)

# 3. Gauss filtri
# (kernel_size, kernel_size), sigmaX (x yo'nalishi bo'yicha standart og'ish)
gaussian_filtered_img = cv2.GaussianBlur(img_braille_orig, (5,5), 0) # 0 avtomatik hisoblashni bildiradi

# Natijalarni ko'rsatish
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_braille_orig, cmap='gray')
plt.title('Asl (Shovqinli) Tasvir')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(mean_filtered_img, cmap='gray')
plt.title('O\'rtacha Filtr')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(median_filtered_img, cmap='gray')
plt.title('Median Filtr')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gaussian_filtered_img, cmap='gray')
plt.title('Gauss Filtr')
plt.axis('off')

plt.tight_layout()
plt.show()

# Kichik bir amaliy mashq:
# Asl tasvirning markazidagi 5x5 blokni (piksel qiymatlarini)
# va ushbu blokning o'rtacha, median, Gauss filtridan keyingi qiymatlarini solishtiring.
# Buni o'zingiz kod yozib amalga oshiring.