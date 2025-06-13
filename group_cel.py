import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Tasvirni yuklash (binarizatsiya qilingan Brayl tasvirini ishlatamiz)
try:
    img_braille_binary = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img_braille_binary is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Xato: 'braille_binary_output.png' fayli topilmadi.")
    # Test uchun sun'iy binarizatsiya qilingan tasvir yaratamiz
    img_braille_binary = np.full((250, 400), 255, dtype=np.uint8) # Oq fon

    # Birinchi Brayl katakchasi (simulyatsiya)
    cv2.circle(img_braille_binary, (50, 50), 8, 0, -1) # 1
    cv2.circle(img_braille_binary, (50, 90), 8, 0, -1) # 2
    cv2.circle(img_braille_binary, (50, 130), 8, 0, -1) # 3 (bo'sh)
    cv2.circle(img_braille_binary, (90, 50), 8, 0, -1) # 4
    cv2.circle(img_braille_binary, (90, 90), 8, 0, -1) # 5
    cv2.circle(img_braille_binary, (90, 130), 8, 0, -1) # 6 (bo'sh)

    # Ikkinchi Brayl katakchasi (simulyatsiya)
    cv2.circle(img_braille_binary, (180, 50), 8, 0, -1) # 1
    cv2.circle(img_braille_binary, (180, 90), 8, 0, -1) # 2 (bo'sh)
    cv2.circle(img_braille_binary, (180, 130), 8, 0, -1) # 3
    cv2.circle(img_braille_binary, (220, 50), 8, 0, -1) # 4 (bo'sh)
    cv2.circle(img_braille_binary, (220, 90), 8, 0, -1) # 5
    cv2.circle(img_braille_binary, (220, 130), 8, 0, -1) # 6

    # Tasodifiy shovqin
    for _ in range(50):
        y = np.random.randint(0, img_braille_binary.shape[0])
        x = np.random.randint(0, img_braille_binary.shape[1])
        cv2.circle(img_braille_binary, (x, y), np.random.randint(2, 5), 0, -1)

    print("Test uchun sun'iy binarizatsiya qilingan tasvir yaratildi.")

# --- Avvalgi bosqichdan olingan Brayl nuqtalari xususiyatlari ---
# Quyidagi kodni avvalgi 'Xususiyatlarni ajratish' bosqichidan olingan 'braille_dots' ro'yxati bilan almashtiring.
# Bu yerda biz tasodifiy ma'lumotlar yaratamiz.
if 'braille_dots' not in locals() or not braille_dots: # Agar braille_dots bo'lmasa yoki bo'sh bo'lsa
    # Konturlarni topish va xususiyatlarini ajratish (oldingi qadamdan takror)
    contours, _ = cv2.findContours(img_braille_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    braille_dots = []
    min_dot_area, max_dot_area = 20, 150 # Bu qiymatlarni o'zingizning braille rasmizga moslang
    min_circularity = 0.6
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            continue # Area 0 bo'lsa tashlab yuboramiz

        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        if min_dot_area < area < max_dot_area and circularity > min_circularity:
            braille_dots.append({'center': (cx, cy), 'area': area, 'circularity': circularity})

if not braille_dots:
    print("Brayl nuqtalari topilmadi. Guruhlashni boshlab bo'lmaydi.")
else:
    # --- Nuqtalarni guruhlash (DBSCAN) ---
    # Topilgan nuqtalarning markaziy koordinatalarini NumPy massiviga aylantiramiz
    X = np.array([dot['center'] for dot in braille_dots])

    # DBSCAN parametrlari
    # eps: Klasterdagi nuqtalar orasidagi maksimal masofa.
    #      Bu Brayl katakchasidagi eng yaqin nuqtalar orasidagi masofaga qarab sozlanadi.
    #      Taxminiy vertikal yoki gorizontal nuqta masofasidan biroz kattaroq.
    #      (Masalan, sizning Brayl tasviringizda nuqtalar markazi orasida 40 piksel bo'lsa, 50-60 ni bering)
    eps_val = 50 # Misol uchun, bu qiymatni o'zgartirishingiz kerak
    # min_samples: Klaster hosil qilish uchun kerak bo'lgan minimal nuqtalar soni.
    #              Brayl katakchasida kamida bitta nuqta bo'lishi mumkin.
    min_samples_val = 1

    db = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X)
    labels = db.labels_

    # Klasterlarni qayta ishlash
    unique_labels = set(labels)
    braille_cells = []
    colors = plt.cm.get_cmap('tab10', len(unique_labels)) # Har bir klasterga rang berish

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_braille_binary, cmap='gray')
    ax.set_title('Guruhlangan Brayl Katakchalari')
    ax.axis('off')

    for k, col in zip(unique_labels, colors.colors):
        if k == -1: # Shovqin nuqtalari (klasterga kirmaganlar)
            continue

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        # Har bir klaster uchun barcha nuqtalarni chizish
        ax.scatter(xy[:, 0], xy[:, 1], s=100, c=[col], alpha=0.7, edgecolors='k')

        # Klasterdagi nuqtalarni Brayl katakchasiga joylashtirish
        if len(xy) > 0:
            # Nuqtalarning eng chap va eng yuqori qismini topish
            min_x, min_y = np.min(xy[:, 0]), np.min(xy[:, 1])

            # Klasterdagi nuqtalarni nisbiy pozitsiyasiga ko'ra saralash
            # va Brayl matritsasiga joylashtirish
            cell_dots_relative = []
            # Brayl katakchasidagi nuqtalar orasidagi gorizontal va vertikal o'rtacha masofani taxmin qilish
            # Sizning tasviringizga qarab bu qiymatlarni aniqlashingiz kerak.
            # Masalan, 40 piksel
            avg_h_dist = 40 # Gorizontal masofa
            avg_v_dist = 40 # Vertikal masofa

            # 6-nuqtali Brayl katakchasi uchun bo'sh massiv
            # Index 0:1-nuqta, 1:2-nuqta, ..., 5:6-nuqta
            braille_cell_matrix = [0] * 6

            for dot_x, dot_y in xy:
                # Nuqtaning katakcha ichidagi nisbiy pozitsiyasini hisoblash
                # Eng yaqin standart pozitsiyani topish
                col_idx = round((dot_x - min_x) / avg_h_dist)
                row_idx = round((dot_y - min_y) / avg_v_dist)

                # 6-nuqtali Brayl pozitsiyasini hisoblash (0-5 oralig'ida)
                # 0-ustun (chap): 0, 1, 2 (1,2,3-nuqta)
                # 1-ustun (o'ng): 3, 4, 5 (4,5,6-nuqta)
                braille_pos = -1
                if col_idx == 0: # Chap ustun
                    if row_idx == 0: braille_pos = 0 # 1-nuqta
                    elif row_idx == 1: braille_pos = 1 # 2-nuqta
                    elif row_idx == 2: braille_pos = 2 # 3-nuqta
                elif col_idx == 1: # O'ng ustun
                    if row_idx == 0: braille_pos = 3 # 4-nuqta
                    elif row_idx == 1: braille_pos = 4 # 5-nuqta
                    elif row_idx == 2: braille_pos = 5 # 6-nuqta

                if 0 <= braille_pos < 6:
                    braille_cell_matrix[braille_pos] = 1 # Nuqta mavjud

            braille_cells.append(braille_cell_matrix)

            # Katakcha atrofida to'g'rito'rtburchak chizish
            max_x, max_y = np.max(xy[:, 0]), np.max(xy[:, 1])
            ax.add_patch(plt.Rectangle((min_x - 5, min_y - 5), (max_x - min_x) + 10, (max_y - min_y) + 10,
                                        fill=False, edgecolor=col, linewidth=2))

    plt.show()

    print("\nTopilgan Brayl katakchalari (6-nuqtali matritsa sifatida, 1=nuqta, 0=bo'sh):")
    for cell_matrix in braille_cells:
        print(cell_matrix)