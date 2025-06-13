import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import io

# ==============================================================================
# 0. BRAILLE KOD LUG'ATI
# Bu yerda siz barcha Brayl kodlarini o'z ichiga olgan to'liq lug'atni yaratishingiz kerak.
# Hozirda faqat misol uchun ba'zi belgilar kiritilgan.
# Agar 8-nuqtali Brayl bo'lsa, har bir tuple 8 ta elementdan iborat bo'lishi kerak.
# ==============================================================================
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
    (0, 0, 1, 1, 0, 1): ':',
    (0, 0, 1, 1, 1, 0): '.',
    (0, 1, 0, 0, 0, 0): ',',
    (0, 0, 1, 0, 0, 0): "'",
    (0, 0, 1, 1, 1, 1): ';',
    (0, 1, 1, 0, 0, 0): '-',
    (0, 0, 0, 1, 0, 1): '(',
    (0, 0, 0, 1, 1, 1): '!',
    (0, 0, 1, 0, 1, 1): 'NUMBER_SIGN', # Raqam belgisi
    (0, 0, 0, 0, 0, 1): 'CAPITAL_SIGN', # Bosh harf belgisi
}

# Raqamlar lug'ati (NUMBER_SIGN dan keyin keladigan harflar uchun)
NUMBER_MAP = {
    'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5',
    'F': '6', 'G': '7', 'H': '8', 'I': '9', 'J': '0'
}


# ==============================================================================
# 1. YORDAMCHI FUNKSIYALAR (Image Processing Steps)
# ==============================================================================

def get_empty_image(original_image=None):
    """Bo'sh (qora) tasvirni qaytaradi, original tasvirning o'lchamida."""
    if original_image is not None and original_image.shape:
        # Agar rangli bo'lsa, kulrang qilish uchun bitta kanalni tanlaymiz
        if len(original_image.shape) == 3:
            return np.zeros(original_image.shape[:2], dtype=np.uint8)
        return np.zeros(original_image.shape, dtype=np.uint8)
    return np.zeros((100, 100), dtype=np.uint8) # Agar original image bo'lmasa, standart o'lcham


@st.cache_data
def apply_denoising(image, method, params):
    """Tasvirga shovqinni tozalash filtrini qo'llaydi."""
    if image is None:
        return get_empty_image() # None o'rniga bo'sh tasvir qaytaramiz
    
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    denoised_img = image_gray
    try: # Xatolarni ushlash uchun try-except bloki
        if method == "Median Filtr":
            ksize = params.get('ksize', 5)
            denoised_img = cv2.medianBlur(image_gray, ksize)
        elif method == "Gauss Filtr":
            ksize = params.get('ksize', 5)
            sigma = params.get('sigma', 0)
            denoised_img = cv2.GaussianBlur(image_gray, (ksize, ksize), sigma)
        elif method == "Bilateral Filtr":
            d = params.get('d', 9)
            sigmaColor = params.get('sigmaColor', 75)
            sigmaSpace = params.get('sigmaSpace', 75)
            denoised_img = cv2.bilateralFilter(image_gray, d, sigmaColor, sigmaSpace)
    except Exception as e:
        st.error(f"Shovqinni tozalashda xato yuz berdi: {e}")
        return get_empty_image(image) # Xato bo'lsa bo'sh tasvir qaytarish
    return denoised_img

@st.cache_data
def apply_contrast_enhancement(image, method, params):
    """Tasvirga kontrastni oshirish usulini qo'llaydi."""
    if image is None:
        return get_empty_image()
        
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    enhanced_img = image_gray
    try:
        if method == "Histogram Tenglashtirish":
            enhanced_img = cv2.equalizeHist(image_gray)
        elif method == "CLAHE":
            clipLimit = params.get('clipLimit', 2.0)
            tileGridSize = params.get('tileGridSize', (8, 8))
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            enhanced_img = clahe.apply(image_gray)
    except Exception as e:
        st.error(f"Kontrastni oshirishda xato yuz berdi: {e}")
        return get_empty_image(image)
    return enhanced_img

@st.cache_data
def apply_binarization(image, method, params):
    """Tasvirga binarizatsiya usulini qo'llaydi."""
    if image is None:
        return get_empty_image()

    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    binary_img = image_gray
    try:
        if method == "Otsu Binarizatsiya":
            _, binary_img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Adaptiv Gauss Binarizatsiya":
            blockSize = params.get('blockSize', 11)
            C = params.get('C', 2)
            if blockSize % 2 == 0:
                blockSize += 1
            binary_img = cv2.adaptiveThreshold(image_gray, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, blockSize, C)
    except Exception as e:
        st.error(f"Binarizatsiyada xato yuz berdi: {e}")
        return get_empty_image(image)
    return binary_img

@st.cache_data
def extract_and_group_dots(binary_image, dot_params, grouping_params):
    """Binarizatsiya qilingan tasvirdan Brayl nuqtalarini ajratadi va guruhlaydi."""
    if binary_image is None or binary_image.shape[0] == 0 or binary_image.shape[1] == 0:
        return [], get_empty_image() # None o'rniga bo'sh tasvir qaytaramiz
    
    img_viz = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR) # Vizualizatsiya uchun nusxa
    
    braille_dots = []
    detected_braille_cells_raw = []

    try:
        # Konturlarni topish
        contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_dot_area = dot_params.get('min_dot_area', 20)
        max_dot_area = dot_params.get('max_dot_area', 150)
        min_circularity = dot_params.get('min_circularity', 0.6)

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            M = cv2.moments(contour)
            
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                continue

            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            if min_dot_area < area < max_dot_area and circularity > min_circularity:
                braille_dots.append({'center': (cx, cy), 'area': area, 'circularity': circularity})

        if not braille_dots:
            return [], img_viz # Nuqtalar topilmasa ham, rasm bo'sh bo'lmasin

        # DBSCAN yordamida nuqtalarni guruhlash
        X = np.array([dot['center'] for dot in braille_dots])
        
        eps_val = grouping_params.get('eps', 50)
        min_samples_val = grouping_params.get('min_samples', 1)

        db = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X)
        labels = db.labels_

        unique_labels = set(labels)
        
        avg_h_dist = grouping_params.get('avg_h_dist', 40)
        avg_v_dist = grouping_params.get('avg_v_dist', 40)

        for k in unique_labels:
            if k == -1: # Shovqin nuqtalari
                continue

            class_member_mask = (labels == k)
            xy = X[class_member_mask]

            if len(xy) > 0:
                min_x, min_y = np.min(xy[:, 0]), np.min(xy[:, 1])
                
                braille_cell_matrix = [0] * 6 # 6-nuqtali Brayl uchun shablon

                for dot_x, dot_y in xy:
                    col_idx = round((dot_x - min_x) / avg_h_dist)
                    row_idx = round((dot_y - min_y) / avg_v_dist)

                    braille_pos = -1
                    if col_idx == 0:
                        if row_idx == 0: braille_pos = 0
                        elif row_idx == 1: braille_pos = 1
                        elif row_idx == 2: braille_pos = 2
                    elif col_idx == 1:
                        if row_idx == 0: braille_pos = 3
                        elif row_idx == 1: braille_pos = 4
                        elif row_idx == 2: braille_pos = 5
                    
                    if 0 <= braille_pos < 6:
                        braille_cell_matrix[braille_pos] = 1
                
                if 1 <= sum(braille_cell_matrix) <= 6:
                    detected_braille_cells_raw.append({
                        'matrix': braille_cell_matrix,
                        'center_x': np.mean(xy[:, 0]),
                        'center_y': np.mean(xy[:, 1])
                    })

        detected_braille_cells = sorted(detected_braille_cells_raw, key=lambda d: (d['center_y'], d['center_x']))
        
        # Vizualizatsiya
        for i, cell in enumerate(detected_braille_cells):
            center_x, center_y = int(cell['center_x']), int(cell['center_y'])
            
            for dot_idx, dot_val in enumerate(cell['matrix']):
                if dot_val == 1:
                    dot_cx = int(center_x - (avg_h_dist / 2) + (avg_h_dist * (dot_idx // 3)))
                    dot_cy = int(center_y - (avg_v_dist / 2) + (avg_v_dist * (dot_idx % 3)))
                    cv2.circle(img_viz, (dot_cx, dot_cy), 5, (0, 255, 0), -1)

            cell_rect_x = int(center_x - avg_h_dist)
            cell_rect_y = int(center_y - avg_v_dist)
            cell_rect_w = int(avg_h_dist * 2)
            cell_rect_h = int(avg_v_dist * 3)
            
            cv2.rectangle(img_viz, (cell_rect_x, cell_rect_y), 
                          (cell_rect_x + cell_rect_w, cell_rect_y + cell_rect_h), 
                          (255, 0, 0), 2)
    except Exception as e:
        st.error(f"Nuqtalarni ajratish yoki guruhlashda xato yuz berdi: {e}")
        return [], get_empty_image(binary_image)

    return detected_braille_cells, img_viz

@st.cache_data
def recognize_braille_cells(braille_cells_data, braille_code_dict):
    """Guruhlangan Brayl katakchalarini belgilariga aylantiradi."""
    recognized_chars = []
    for cell in braille_cells_data:
        cell_pattern_tuple = tuple(cell['matrix'])
        recognized_char = braille_code_dict.get(cell_pattern_tuple, '?')
        recognized_chars.append(recognized_char)
    return recognized_chars

def post_process_text(recognized_chars, number_map):
    """Raqamlar va bosh harflar belgilarini qayta ishlaydi."""
    processed_text = []
    is_number_mode = False
    is_capital_mode = False

    i = 0
    while i < len(recognized_chars):
        char = recognized_chars[i]
        if char == 'NUMBER_SIGN':
            is_number_mode = True
            i += 1
            continue
        elif char == 'CAPITAL_SIGN':
            is_capital_mode = True
            i += 1
            continue

        if is_number_mode:
            processed_text.append(number_map.get(char, char))
            is_number_mode = False
        elif is_capital_mode:
            processed_text.append(char.upper())
            is_capital_mode = False
        else:
            processed_text.append(char.lower())
        i += 1
    return "".join(processed_text)

# ==============================================================================
# 2. STREAMLIT ILK UZGARMALARI VA SOZLAMALARI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Brayl Tasviridan Matnni Tanib Olish (OCR)")

st.title("Brayl tasviridan matnni tanib olish (BTMTO) 🇧🇷📖")
st.markdown("### Tasvirni yuklang va Brayl matnini tanib olish uchun jarayon parametrlarini sozlang.")

# Foydalanuvchi tasvirni yuklash
uploaded_file = st.file_uploader("Brayl tasvirini yuklang...", type=["png", "jpg", "jpeg", "bmp"])

# ==============================================================================
# 3. YON PANEL (Sidebar) - Parametrlar va Bosqichlar Tanlash
# ==============================================================================
st.sidebar.header("Parametrlarni sozlash")

# Denoising sozlamalari
st.sidebar.subheader("1. Shovqinni tozalash")
denoising_method = st.sidebar.selectbox(
    "Filtr usulini tanlang:",
    ("Yo'q", "Median Filtr", "Gauss Filtr", "Bilateral Filtr"),
    index=3
)
denoising_params = {}
if denoising_method == "Median Filtr":
    denoising_params['ksize'] = st.sidebar.slider("Median Kichikligi (ksize)", 3, 15, 5, step=2)
elif denoising_method == "Gauss Filtr":
    denoising_params['ksize'] = st.sidebar.slider("Gauss Kichikligi (ksize)", 3, 15, 5, step=2)
    denoising_params['sigma'] = st.sidebar.slider("Gauss Sigma", 0.0, 10.0, 0.0, step=0.1)
elif denoising_method == "Bilateral Filtr":
    denoising_params['d'] = st.sidebar.slider("Bilateral Diametr (d)", 1, 20, 9)
    denoising_params['sigmaColor'] = st.sidebar.slider("Bilateral Sigma Color", 1, 200, 75)
    denoising_params['sigmaSpace'] = st.sidebar.slider("Bilateral Sigma Space", 1, 200, 75)

# Kontrastni oshirish sozlamalari
st.sidebar.subheader("2. Kontrastni oshirish")
contrast_method = st.sidebar.selectbox(
    "Kontrast usulini tanlang:",
    ("Yo'q", "Histogram Tenglashtirish", "CLAHE"),
    index=2
)
contrast_params = {}
if contrast_method == "CLAHE":
    contrast_params['clipLimit'] = st.sidebar.slider("CLAHE Chegarasi (clipLimit)", 0.5, 10.0, 2.0, step=0.1)
    tile_size_val = st.sidebar.slider("CLAHE Panjara O'lchami (tileGridSize)", 4, 32, 8, step=2)
    contrast_params['tileGridSize'] = (tile_size_val, tile_size_val)

# Binarizatsiya sozlamalari
st.sidebar.subheader("3. Binarizatsiya")
binarization_method = st.sidebar.selectbox(
    "Binarizatsiya usulini tanlang:",
    ("Otsu Binarizatsiya", "Adaptiv Gauss Binarizatsiya"),
    index=1
)
binarization_params = {}
if binarization_method == "Adaptiv Gauss Binarizatsiya":
    binarization_params['blockSize'] = st.sidebar.slider("Adaptiv Blok O'lchami (blockSize)", 3, 51, 11, step=2)
    binarization_params['C'] = st.sidebar.slider("Adaptiv C (Offset)", -10, 10, 2)

# Xususiyatlarni ajratish va guruhlash sozlamalari
st.sidebar.subheader("4. Nuqtalarni ajratish va guruhlash")
dot_params = {
    'min_dot_area': st.sidebar.slider("Min Nuqta Maydoni (piksel)", 1, 500, 20),
    'max_dot_area': st.sidebar.slider("Max Nuqta Maydoni (piksel)", 1, 1000, 150),
    'min_circularity': st.sidebar.slider("Min Doiraviylik (0-1)", 0.0, 1.0, 0.6, step=0.05)
}
grouping_params = {
    'eps': st.sidebar.slider("DBSCAN Eps (Nuqta Guruh Masofasi)", 10, 100, 50),
    'min_samples': st.sidebar.slider("DBSCAN Min Samples (Klaster uchun)", 1, 10, 1),
    'avg_h_dist': st.sidebar.slider("O'rtacha Gorizontal Nuqta Masofasi", 10, 100, 40),
    'avg_v_dist': st.sidebar.slider("O'rtacha Vertikal Nuqta Masofasi", 10, 100, 40)
}

# ==============================================================================
# 4. TASVIRNI QAYTA ISHLASH FUNKSIYASI
# ==============================================================================
def process_braille_image(image_data):
    if image_data is None:
        st.warning("Iltimos, tasvir yuklang.")
        return

    # Tasvirni o'qish (NumPy massiviga aylantirish)
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Natijalarni ko'rsatish uchun joy ajratish
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Asl tasvir")
        st.image(original_image, channels="BGR", use_container_width=True) # use_container_width

    # 1. Shovqinni tozalash
    denoised_image = apply_denoising(original_image, denoising_method, denoising_params)
    with col2:
        st.subheader(f"1. Shovqinni tozalash ({denoising_method})")
        st.image(denoised_image, use_container_width=True) # use_container_width

    # 2. Kontrastni oshirish
    enhanced_image = apply_contrast_enhancement(denoised_image, contrast_method, contrast_params)
    with col1:
        st.subheader(f"2. Kontrastni oshirish ({contrast_method})")
        st.image(enhanced_image, use_container_width=True) # use_container_width

    # 3. Binarizatsiya
    binary_image = apply_binarization(enhanced_image, binarization_method, binarization_params)
    with col2:
        st.subheader(f"3. Binarizatsiya ({binarization_method})")
        st.image(binary_image, use_container_width=True) # use_container_width

    # 4. Xususiyatlarni ajratish va guruhlash
    detected_braille_cells, img_grouped_viz = extract_and_group_dots(binary_image, dot_params, grouping_params)
    
    with col1:
        st.subheader("4. Guruhlangan brayl katakchalari")
        st.image(img_grouped_viz, channels="BGR", use_container_width=True) # use_container_width
    
    # 5. Belgilarni tanib olish
    recognized_chars = recognize_braille_cells(detected_braille_cells, BRAILLE_CODE)
    
    # 6. Post-processing (raqamlar va bosh harflar uchun)
    final_text = post_process_text(recognized_chars, NUMBER_MAP)

    with col2:
        st.subheader("5. Tanib olingan matn")
        st.write(f"**Xom tanib olingan matn:** {''.join(recognized_chars)}")
        st.write(f"**Qayta ishlangan matn:** {final_text}")
        
        # Vizualizatsiya: Tanib olingan belgilar rasmi
        # Agar binary_image None bo'lsa, xato beradi, shuning uchun tekshiramiz
        if binary_image is not None and binary_image.shape[0] > 0 and binary_image.shape[1] > 0:
            img_final_recognition_viz = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
            for cell in detected_braille_cells:
                cell_pattern_tuple = tuple(cell['matrix'])
                recognized_char = BRAILLE_CODE.get(cell_pattern_tuple, '?')
                
                text_x, text_y = int(cell['center_x']), int(cell['center_y'] + 20)
                cv2.putText(img_final_recognition_viz, recognized_char, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                cell_rect_x = int(cell['center_x'] - grouping_params['avg_h_dist'])
                cell_rect_y = int(cell['center_y'] - grouping_params['avg_v_dist'])
                cell_rect_w = int(grouping_params['avg_h_dist'] * 2)
                cell_rect_h = int(grouping_params['avg_v_dist'] * 3)
                cv2.rectangle(img_final_recognition_viz, (cell_rect_x, cell_rect_y), 
                              (cell_rect_x + cell_rect_w, cell_rect_y + cell_rect_h), 
                              (255, 0, 0), 2)
            
            st.subheader("6. Tanib olingan belgilar vizualizatsiyasi")
            st.image(img_final_recognition_viz, channels="BGR", use_container_width=True) # use_container_width
        else:
            st.warning("Tanib olingan belgilar vizualizatsiyasi uchun binarizatsiya qilingan tasvir mavjud emas.")


# ==============================================================================
# 5. STREAMLIT ASOSIY ISHLASH OQIMI
# ==============================================================================
if uploaded_file is not None:
    process_braille_image(uploaded_file)
else:
    st.info("Boshlash uchun brayl tasvirini yuklang.")
    st.markdown("""
        **Eslatma:**
        * Eng yaxshi natijalar uchun yuqori aniqlikdagi va yaxshi yoritilgan Brayl tasvirlarini yuklang.
        * Yon paneldagi (sidebar) parametrlarni tasviringizga mos ravishda sozlang.
            * **Nuqta maydoni (Area)**: Brayl nuqtalarining o'rtacha piksel maydonini hisobga oling.
            * **Doiraviylik (Circularity)**: Nuqtalarning dumaloqligi uchun chegara.
            * **DBSCAN Eps**: Bir Brayl katakchasidagi eng uzoq nuqtalar orasidagi masofadan biroz kattaroq qilib belgilang.
            * **O'rtacha Gorizontal/Vertikal Masofa**: Bu Brayl katakchasidagi nuqtalar orasidagi o'rtacha masofa. U nuqtalarning katakchadagi aniq pozitsiyasini aniqlash uchun ishlatiladi.
        * Hozircha, kod 6-nuqtali Brayl uchun optimallashtirilgan.
    """)