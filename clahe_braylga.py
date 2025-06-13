import cv2
import numpy as np
import matplotlib.pyplot as plt

# Brayl tasvirini yuklash (past kontrastli rasm bilan test qiling)
try:
    img_braille = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Rasmni yuklashda xato! 'braille_sample_low_contrast.png' fayli topilmadi.")
    # Test uchun sun'iy, past kontrastli va shovqinli Brayl tasvirini yaratamiz
    img_braille = np.zeros((150, 250), dtype=np.uint8)
    img_braille[30:40, 30:40] = 180; img_braille[30:40, 60:70] = 190
    img_braille[50:60, 30:40] = 170; img_braille[50:60, 60:70] = 185
    img_braille[70:80, 45:55] = 195
    img_braille = img_braille + 80 # Fonga kulrang tus
    img_braille = np.clip(img_braille + np.random.normal(0, 20, img_braille.shape), 0, 255).astype(np.uint8) # Shovqin
    print("Test uchun sun'iy, past kontrastli Brayl tasviri yaratildi.")


# CLAHE obyektini yaratish
# clipLimit: Kontrast chegarasi. Qanchalik katta bo'lsa, kontrast shunchalik kuchli bo'ladi.
#            Shovqinli tasvirlar uchun pastroq qiymat (masalan, 1.0-3.0) tavsiya etiladi.
#            Detal talab qilinadigan joylarda yuqoriroq (masalan, 4.0-8.0) ishlatilishi mumkin.
# tileGridSize: Tasvirni bo'lish uchun panjara o'lchami (masalan, 8x8, 16x16, 32x32).
#               Kichikroq panjara lokalroq kontrastni boshqaradi, ammo shovqinni kuchaytirishi mumkin.
#               Brayl nuqtalari uchun nuqtalarning o'lchamiga mos keladigan o'lcham tanlash muhim.
#               Agar nuqtalar katta bo'lsa, kattaroq panjara.
clahe_config = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Optimal qiymatlar

# CLAHE ni qo'llash
clahe_output = clahe_config.apply(img_braille)

# Natijalarni ko'rsatish
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_braille, cmap='gray')
plt.title('Asl Tasvir (past kontrast)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(clahe_output, cmap='gray')
plt.title(f'CLAHE qo\'llandi (clipLimit=2.0, tileGridSize=(8,8))')
plt.axis('off')

plt.tight_layout()
plt.show()

# Turli parametrlarni sinab ko'ring
# clahe_config_high_clip = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
# clahe_output_high_clip = clahe_config_high_clip.apply(img_braille)
# cv2.imshow('CLAHE (clipLimit=5.0)', clahe_output_high_clip)

# clahe_config_small_tile = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
# clahe_output_small_tile = clahe_config_small_tile.apply(img_braille)
# cv2.imshow('CLAHE (tileGridSize=(4,4))', clahe_output_small_tile)

# cv2.waitKey(0)
# cv2.destroyAllWindows()