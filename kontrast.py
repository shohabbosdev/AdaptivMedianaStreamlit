import cv2
import numpy as np
import matplotlib.pyplot as plt

# Brayl tasvirini yuklash
try:
    # Bu yerda past kontrastli Brayl tasvirini ishlatish tavsiya etiladi
    img_braille_orig = cv2.imread('image copy.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_orig is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Rasmni yuklashda xato! 'braille_sample_low_contrast.png' fayli topilmadi.")
    # Agar rasm topilmasa, test uchun past kontrastli sun'iy rasm yaratamiz
    img_braille_orig = np.zeros((150, 250), dtype=np.uint8)
    # Ba'zi "nuqtalarni" past kontrastda qo'shamiz
    img_braille_orig[30:40, 30:40] = 180
    img_braille_orig[30:40, 60:70] = 190
    img_braille_orig[50:60, 30:40] = 170
    img_braille_orig[50:60, 60:70] = 185
    img_braille_orig[70:80, 45:55] = 195
    # Fonga biroz kulrang tus beramiz
    img_braille_orig = img_braille_orig + 100
    img_braille_orig = np.clip(img_braille_orig, 0, 255).astype(np.uint8)
    # Shovqin qo'shamiz
    mean = 0
    var = 100
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (img_braille_orig.shape[0], img_braille_orig.shape[1]))
    img_braille_orig = img_braille_orig + gauss
    img_braille_orig = np.clip(img_braille_orig, 0, 255).astype(np.uint8)
    print("Test uchun past kontrastli sun'iy Brayl tasviri yaratildi.")


# 1. Oddiy histogram tenglashtirish
equalized_img = cv2.equalizeHist(img_braille_orig)

# 2. CLAHE (Cheklangan kontrastli adaptiv histogram tenglashtirish)
# CLAHE obyektini yaratish
# clipLimit: Kontrastni cheklash parametri. Katta qiymat ko'proq kontrast beradi, lekin shovqinni kuchaytirishi mumkin.
# tileGridSize: Tasvirni bo'lish uchun panjara o'lchami (masalan, 8x8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img_braille_orig)

# Natijalarni ko'rsatish
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_braille_orig, cmap='gray')
plt.title('Asl (Past Kontrastli) Tasvir')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Histogram Tenglashtirish')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(clahe_img, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

plt.tight_layout()
plt.show()

# Kichik bir amaliy mashq:
# Asl tasvirning histogrammasini va tenglashtirilgan tasvirlarning histogrammasini chizib ko'ring.
# cv2.calcHist funksiyasidan foydalanishingiz mumkin.
# Bu sizga piksel qiymatlarining qanday taqsimlanishini ko'rsatadi.