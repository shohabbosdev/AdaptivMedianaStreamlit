import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# --- Avvalgi bosqichlardan olingan Brayl tasviri va nuqtalari ---
# Binarizatsiya qilingan tasvirni yuklash
try:
    img_braille_binary = cv2.imread('braille_binary_output.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_binary is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Xato: 'braille_binary_output.png' fayli topilmadi.")
    # Test uchun sun'iy binarizatsiya qilingan tasvir yaratamiz (oldingi misoldan takror)
    img_braille_binary = np.full((250, 400), 255, dtype=np.uint8) # Oq fon
    # Birinchi Brayl katakchasi: 'A' (100000)
    cv2.circle(img_braille_binary, (50, 50), 8, 0, -1) # 1-nuqta
    # Ikkinchi Brayl katakchasi: 'B' (110000)
    cv2.circle(img_braille_binary, (150, 50), 8, 0, -1) # 1-nuqta
    cv2.circle(img_braille_binary, (150, 90), 8, 0, -1) # 2-nuqta
    # Uchinchi Brayl katakchasi: 'L' (101010)
    cv2.circle(img_braille_binary, (250, 50), 8, 0, -1) # 1-nuqta
    cv2.circle(img_braille_binary, (250, 130), 8, 0, -1) # 3-nuqta
    cv2.circle(img_braille_binary, (290, 90), 8, 0, -1) # 5-nuqta
    print("Test uchun sun'iy binarizatsiya qilingan tasvir yaratildi.")

# Brayl nuqtalarini ajratish (oldingi qadamdan takror, to'liq ishga tushirish uchun)
contours, _ = cv2.findContours(img_braille_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
braille_dots = []
min_dot_area, max_dot_area = 20, 150
min_circularity = 0.6
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        continue
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    if min_dot_area < area < max_dot_area and circularity > min_circularity:
        braille_dots.append({'center': (cx, cy), 'area': area, 'circularity': circularity})

# Nuqtalarni guruhlash (oldingi qadamdan takror)
if not braille_dots:
    print("Brayl nuqtalari topilmadi. Tanib olishni boshlab bo'lmaydi.")
    exit() # Nuqtalarsiz davom etmaymiz

X = np.array([dot['center'] for dot in braille_dots])
eps_val = 50 # Nuqtalar orasidagi masofa
min_samples_val = 1
db = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X)
labels = db.labels_

unique_labels = set(labels)
detected_braille_cells_raw = [] # Guruhlangan nuqtalarni saqlash uchun

avg_h_dist = 40 # Gorizontal nuqta masofasi (pikselda)
avg_v_dist = 40 # Vertikal nuqta masofasi (pikselda)

for k in unique_labels:
    if k == -1: # Shovqin
        continue
    class_member_mask = (labels == k)
    xy = X[class_member_mask]

    if len(xy) > 0:
        min_x, min_y = np.min(xy[:, 0]), np.min(xy[:, 1])
        
        braille_cell_matrix = [0] * 6 # 6-nuqtali Brayl uchun
        for dot_x, dot_y in xy:
            col_idx = round((dot_x - min_x) / avg_h_dist)
            row_idx = round((dot_y - min_y) / avg_v_dist)

            braille_pos = -1
            if col_idx == 0:
                if row_idx == 0: braille_pos = 0 # 1-nuqta
                elif row_idx == 1: braille_pos = 1 # 2-nuqta
                elif row_idx == 2: braille_pos = 2 # 3-nuqta
            elif col_idx == 1:
                if row_idx == 0: braille_pos = 3 # 4-nuqta
                elif row_idx == 1: braille_pos = 4 # 5-nuqta
                elif row_idx == 2: braille_pos = 5 # 6-nuqta
            
            # Qo'shimcha tekshiruv: agar 8-nuqtali Brayl bo'lsa
            # if col_idx == 0 and row_idx == 3: braille_pos = 6 # 7-nuqta
            # if col_idx == 1 and row_idx == 3: braille_pos = 7 # 8-nuqta

            if 0 <= braille_pos < 6: # Yoki 8, agar 8-nuqtali bo'lsa
                braille_cell_matrix[braille_pos] = 1
        
        # Faqat to'g'ri shaklga ega katakchalarni saqlash (masalan, 1-6 nuqta oralig'ida)
        if 1 <= sum(braille_cell_matrix) <= 6: # Summasi 0 dan katta va 6 dan kichik yoki teng bo'lishi kerak
            detected_braille_cells_raw.append({
                'matrix': braille_cell_matrix,
                'center_x': np.mean(xy[:, 0]), # Katakchaning markaziy X
                'center_y': np.mean(xy[:, 1])  # Katakchaning markaziy Y
            })

# Katakchalarni chapdan o'ngga, keyin yuqoridan pastga saralash
# Bu matnni to'g'ri tartibda olish uchun muhim
# Dastlab Y bo'yicha (qatorlar), so'ngra X bo'yicha (ustunlar) saralaymiz
detected_braille_cells = sorted(detected_braille_cells_raw, key=lambda d: (d['center_y'], d['center_x']))

# --- Brayl kod lug'ati (faqat misol uchun kichik qismi) ---
# Haqiqiy ilovada barcha Brayl kodlarini o'z ichiga olgan to'liq lug'at kerak bo'ladi.
# 6-nuqtali Brayl kodlari
BRAILLE_CODE = {
    (1, 0, 0, 0, 0, 0): 'A',
    (1, 1, 0, 0, 0, 0): 'B',
    (1, 0, 0, 1, 0, 0): 'C',
    (1, 0, 0, 1, 1, 0): 'D',
    (1, 0, 0, 0, 1, 0): 'E',
    (1, 1, 0, 1, 0, 0): 'F',
    (1, 1, 0, 1, 1, 0): 'G',
    (1, 1, 0, 0, 1, 0): 'H',
    (0, 1, 0, 1, 0, 0): 'I',
    (0, 1, 0, 1, 1, 0): 'J',
    (1, 0, 1, 0, 0, 0): 'K',
    (1, 1, 1, 0, 0, 0): 'L',
    (1, 0, 1, 1, 0, 0): 'M',
    (1, 0, 1, 1, 1, 0): 'N',
    (1, 0, 1, 0, 1, 0): 'O',
    (1, 1, 1, 1, 0, 0): 'P',
    (1, 1, 1, 1, 1, 0): 'Q',
    (1, 1, 1, 0, 1, 0): 'R',
    (0, 1, 1, 1, 0, 0): 'S',
    (0, 1, 1, 1, 1, 0): 'T',
    (1, 0, 1, 0, 0, 1): 'U',
    (1, 1, 1, 0, 0, 1): 'V',
    (0, 1, 0, 1, 1, 1): 'W',
    (1, 0, 1, 1, 0, 1): 'X',
    (1, 0, 1, 1, 1, 1): 'Y',
    (1, 0, 1, 0, 1, 1): 'Z',
    (0, 0, 0, 0, 0, 0): ' ', # Bo'sh joy
    (0, 0, 1, 0, 1, 1): 'NUMBER_SIGN', # Raqam belgisi
    (0, 0, 0, 0, 0, 1): 'CAPITAL_SIGN', # Bosh harf belgisi
    # Qo'shimcha belgilar va raqamlar shu yerga qo'shiladi
    # Raqamlar odatda NUMBER_SIGN dan keyin keladi.
    # Masalan, NUMBER_SIGN + A = 1, NUMBER_SIGN + B = 2 va hokazo.
}

recognized_text = []

# Natijalarni vizualizatsiya qilish uchun rasm
img_recognized = cv2.cvtColor(img_braille_binary.copy(), cv2.COLOR_GRAY2BGR)

# Har bir aniqlangan Brayl katakchasini tanib olish
for cell in detected_braille_cells:
    cell_pattern_tuple = tuple(cell['matrix'])
    recognized_char = BRAILLE_CODE.get(cell_pattern_tuple, '?') # Topilmasa '?' belgisini qo'yamiz
    recognized_text.append(recognized_char)

    # Tanib olingan belgini rasm ustiga yozish
    text_x, text_y = int(cell['center_x']), int(cell['center_y'] + 20) # Belgining pastiga yozish
    cv2.putText(img_recognized, recognized_char, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Katakcha atrofida to'g'rito'rtburchak chizish
    cv2.rectangle(img_recognized, 
                  (int(cell['center_x'] - avg_h_dist), int(cell['center_y'] - avg_v_dist)), 
                  (int(cell['center_x'] + avg_h_dist), int(cell['center_y'] + avg_v_dist)), 
                  (255, 0, 0), 1) # Moviy to'g'rito'rtburchak

plt.figure(figsize=(10, 8))
plt.imshow(img_recognized)
plt.title('Tanib Olingan Brayl Belgilari')
plt.axis('off')
plt.tight_layout()
plt.show()

# Yakuniy tanib olingan matnni chop etish
final_text = "".join(recognized_text)
print("\nTanib olingan matn:")
print(final_text)

# Qo'shimcha ishlov berish: Raqamlar va bosh harflar
# Agar matnda 'NUMBER_SIGN' yoki 'CAPITAL_SIGN' kabi belgilar bo'lsa,
# ular keyingi belgilarni mos ravishda raqam yoki bosh harf sifatida interpretatsiya qilish uchun ishlatilishi kerak.
# Bu esa murakkabroq mantiqni talab qiladi va alohida post-processing bosqichi bo'lishi mumkin.
# Masalan:
processed_final_text = []
is_number_mode = False
is_capital_mode = False

i = 0
while i < len(recognized_text):
    char = recognized_text[i]
    if char == 'NUMBER_SIGN':
        is_number_mode = True
        i += 1
        continue
    elif char == 'CAPITAL_SIGN':
        is_capital_mode = True
        i += 1
        continue

    if is_number_mode:
        # Raqamlarni aniqlash lug'ati
        NUMBERS = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5',
                   'F': '6', 'G': '7', 'H': '8', 'I': '9', 'J': '0'}
        processed_final_text.append(NUMBERS.get(char, char))
        is_number_mode = False # Faqat keyingi belgi raqam bo'ladi
    elif is_capital_mode:
        processed_final_text.append(char.upper())
        is_capital_mode = False # Faqat keyingi belgi bosh harf bo'ladi
    else:
        processed_final_text.append(char.lower()) # Qolganlari kichik harf

    i += 1

print("\nQayta ishlangan (raqamlar/bosh harflar bilan) matn:")
print("".join(processed_final_text))