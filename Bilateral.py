import cv2
import numpy as np
import matplotlib.pyplot as plt

# Brayl tasvirini yuklash (o'zingizning fayl nomingizni kiriting)
try:
    img_braille_orig = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_orig is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Rasmni yuklashda xato! 'braille_sample_noisy.png' fayli topilmadi.")
    # Agar rasm topilmasa, test uchun sun'iy rasm yaratamiz va shovqin qo'shamiz
    img_braille_orig = np.zeros((150, 250), dtype=np.uint8)
    # Brayl nuqtalari
    img_braille_orig[30:40, 30:40] = 255
    img_braille_orig[30:40, 60:70] = 255
    img_braille_orig[50:60, 30:40] = 255
    img_braille_orig[50:60, 60:70] = 255
    img_braille_orig[70:80, 45:55] = 255

    # Shovqin qo'shamiz (masalan, tuz va murch shovqini)
    row, col = img_braille_orig.shape
    num_pixels = int(0.05 * row * col) # 5% shovqin
    for _ in range(num_pixels):
        y = np.random.randint(0, row)
        x = np.random.randint(0, col)
        img_braille_orig[y, x] = 255 if np.random.rand() > 0.5 else 0

    # Gauss shovqini ham qo'shamiz
    mean = 0
    var = 500
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    img_braille_orig = img_braille_orig + gauss
    img_braille_orig = np.clip(img_braille_orig, 0, 255).astype(np.uint8)

    print("Test uchun sun'iy shovqinli Brayl tasviri yaratildi.")


# Bilateral filtrni qo'llash
# d: Piksel atrofidagi diametr. Katta qiymat sekinroq, lekin ko'proq shovqinni olib tashlaydi.
# sigmaColor: Rang fazosidagi sigma. Katta qiymat bir-biriga o'xshash ranglarni ko'proq aralashtiradi.
# sigmaSpace: Koordinata fazosidagi sigma. Katta qiymat uzoqroqdagi piksellarga ham ta'sir qiladi.
# Odatda, sigmaColor va sigmaSpace bir xil olinadi.
bilateral_filtered_img = cv2.bilateralFilter(img_braille_orig, d=9, sigmaColor=75, sigmaSpace=75)

# Natijalarni ko'rsatish
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_braille_orig, cmap='gray')
plt.title('Asl (Shovqinli) Tasvir')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(bilateral_filtered_img, cmap='gray')
plt.title('Bilateral Filtr')
plt.axis('off')

plt.tight_layout()
plt.show()