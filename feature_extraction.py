import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tasvirni yuklash (kontrasti oshirilgan va binarizatsiya qilingan Brayl tasvirini ishlatamiz)
try:
    # Oldingi bosqichlardan olingan binarizatsiya qilingan tasvirni yuklang.
    # Bu yerda fon oq (255), ob'ektlar (nuqtalar) qora (0) bo'lishi kerak.
    # Agar sizning binarizatsiyangiz fonni qora, ob'ektni oq qilsa,
    # konturlarni topishdan oldin cv2.bitwise_not(image) bilan invert qilib oling.
    img_braille_binary = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_binary is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Xato: 'braille_binary_output.png' fayli topilmadi.")
    # Test uchun sun'iy binarizatsiya qilingan tasvir yaratamiz
    img_braille_binary = np.full((200, 300), 255, dtype=np.uint8) # Oq fon
    # Bir nechta Brayl nuqtalari
    cv2.circle(img_braille_binary, (50, 50), 10, 0, -1) # Qora doira
    cv2.circle(img_braille_binary, (100, 50), 12, 0, -1)
    cv2.circle(img_braille_binary, (50, 100), 10, 0, -1)
    cv2.circle(img_braille_binary, (100, 100), 10, 0, -1)
    # Kichik shovqin nuqtasi
    cv2.circle(img_braille_binary, (15, 15), 3, 0, -1)
    # Cho'zinchoq shovqin
    cv2.ellipse(img_braille_binary, (200, 80), (20, 10), 0, 0, 360, 0, -1)
    print("Test uchun sun'iy binarizatsiya qilingan tasvir yaratildi.")

# Asl binarizatsiya qilingan tasvirni ko'rsatamiz
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_braille_binary, cmap='gray')
plt.title('Binarizatsiya Qilingan Tasvir')
plt.axis('off')

# Konturlarni topish
# cv2.RETR_EXTERNAL faqat tashqi konturlarni oladi, ichki teshiklarni emas.
# cv2.CHAIN_APPROX_SIMPLE kontur nuqtalarini siqadi.
contours, hierarchy = cv2.findContours(img_braille_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Topilgan konturlarni original tasvir ustiga chizamiz (vizualizatsiya uchun)
img_contours = cv2.cvtColor(img_braille_binary, cv2.COLOR_GRAY2BGR) # Rangli qilish
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2) # Yashil rangda konturlarni chizish

print(f"Topilgan konturlar soni: {len(contours)}")

# Har bir konturni tahlil qilish va xususiyatlarini ajratish
braille_dots = []

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True) # True for closed contour

    # Konturning momentlarini hisoblash (markazni topish uchun)
    M = cv2.moments(contour)
    if M['m00'] != 0: # Agar maydon 0 bo'lmasa
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0 # Maydon 0 bo'lsa, markaz 0,0

    # Minimal chegaralovchi to'g'rito'rtburchak
    x, y, w, h = cv2.boundingRect(contour)

    # Doiraviylikni hisoblash (Circularuty)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # Misol uchun, Brayl nuqtasini ajratish shartlari:
    # 1. Maydonning ma'lum bir diapazonda bo'lishi (kichik shovqinni va katta fon obyektlarini chiqarib tashlash)
    # 2. Doiraviylikning ma'lum bir darajada bo'lishi (cho'zinchoq obyektlarni chiqarib tashlash)
    min_dot_area = 5 # Misol uchun, bu qiymatlarni sizning tasvirlaringizga qarab sozlang
    max_dot_area = 40
    min_circularity = 0.7 # Doira uchun 1 ga yaqin, bu esa 0.7 dan yuqori bo'lsin.

    if min_dot_area < area < max_dot_area and circularity > min_circularity:
        braille_dots.append({'center': (cx, cy), 'area': area, 'perimeter': perimeter, 'circularity': circularity, 'bbox': (x,y,w,h)})
        # Topilgan Brayl nuqtasining markazini chizish
        cv2.circle(img_contours, (cx, cy), 5, (255, 0, 255), -1) # Pushti nuqta
        # Chegaralovchi to'g'rito'rtburchakni chizish
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 0, 255), 1) # Qizil to'g'rito'rtburchak

print(f"Filtrlangan Brayl nuqtalari soni: {len(braille_dots)}")

plt.subplot(1, 2, 2)
plt.imshow(img_contours)
plt.title('Topilgan Konturlar va Xususiyatlar')
plt.axis('off')
plt.tight_layout()
plt.show()

# Topilgan nuqtalarning ba'zi xususiyatlarini chop etish
print("\nFiltrlangan Brayl nuqtalarining xususiyatlari:")
for dot in braille_dots:
    print(f"  Markaz: {dot['center']}, Maydon: {dot['area']:.2f}, Doiraviylik: {dot['circularity']:.2f}")