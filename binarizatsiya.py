import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tasvirni yuklash (kontrasti oshirilgan Brayl tasvirini ishlatish tavsiya etiladi)
try:
    img_braille_preprocessed = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_preprocessed is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Rasmni yuklashda xato! 'braille_sample_clahe.png' fayli topilmadi.")
    # Test uchun sun'iy, kontrastli va shovqinli tasvir yaratamiz
    img_braille_preprocessed = np.zeros((150, 250), dtype=np.uint8)
    # Brayl nuqtalari (oq fon ustida qora nuqtalar)
    img_braille_preprocessed[30:40, 30:40] = 50
    img_braille_preprocessed[30:40, 60:70] = 40
    img_braille_preprocessed[50:60, 30:40] = 60
    img_braille_preprocessed[50:60, 60:70] = 45
    img_braille_preprocessed[70:80, 45:55] = 30
    # Fonga oqroq tus beramiz
    img_braille_preprocessed = img_braille_preprocessed + 180
    img_braille_preprocessed = np.clip(img_braille_preprocessed, 0, 255).astype(np.uint8)
    # Tasvirni biroz silliqlaymiz (agar asl rasm juda shovqinli bo'lsa)
    img_braille_preprocessed = cv2.GaussianBlur(img_braille_preprocessed, (5,5), 0)
    print("Test uchun sun'iy, ishlov berilgan Brayl tasviri yaratildi.")


# 1. Global chegara qiymati (Otsu's Binarization)
# cv2.THRESH_BINARY: chegara qiymatidan yuqori piksellar max_val ga, qolganlari 0 ga.
# cv2.THRESH_OTSU: chegara qiymati Otsu usuli bilan avtomatik aniqlanadi.
ret, otsu_binary = cv2.threshold(img_braille_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu usuli bilan aniqlangan chegara qiymati: {ret}")


# 2. Adaptiv chegara qiymati (Adaptive Gaussian Thresholding)
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: chegara qiymati atrofdagi piksellarning Gauss og'irlikli o'rtachasidan hisoblanadi.
# cv2.THRESH_BINARY: binarizatsiya turi.
# blockSize: Piksel atrofiga hisoblash uchun blok o'lchami (toq son bo'lishi kerak, masalan, 11, 15, 21).
#            Brayl nuqtalarining o'lchamidan biroz kattaroq bo'lishi kerak.
# C: O'rtacha qiymatdan ayiriladigan doimiy. Uni o'zgartirib optimal natijaga erishish mumkin.
adaptive_gaussian_binary = cv2.adaptiveThreshold(img_braille_preprocessed, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)


# Natijalarni ko'rsatish
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_braille_preprocessed, cmap='gray')
plt.title('Ishlov berilgan Asl Tasvir')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(otsu_binary, cmap='gray')
plt.title(f'Otsu Binarizatsiya (T={ret:.2f})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(adaptive_gaussian_binary, cmap='gray')
plt.title('Adaptiv Gauss Binarizatsiya')
plt.axis('off')

plt.tight_layout()
plt.show()

# Kichik amaliy mashq:
# Adaptiv Gauss binarizatsiyasidagi `blockSize` va `C` parametrlarini o'zgartirib, natijalar qanday o'zgarishini kuzating.
# Brayl nuqtalari aniq ajralib chiqishi va fon imkon qadar toza bo'lishi uchun bu parametrlarni optimallashtiring.