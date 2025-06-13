import streamlit as st
import cv2
import numpy as np
from PIL import Image # Streamlit'da tasvirlarni yuklash uchun

# Sahifa konfiguratsiyasi
st.set_page_config(
    page_title="Adaptiv Mediana Filtrlash",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Adaptiv Mediana Filtrlash Demonstratsiyasi (Brayl tasviri uchun)")

st.write("""
Ushbu ilova tasvirni yuklash, unga sun'iy 'tuz va murch' shovqini qo'shish
va so'ngra mediana filtri yordamida shovqinni tozalashni namoyish etadi.
Bu, ayniqsa, Brayl tasvirlaridagi nuqta shovqinlarini tozalash uchun foydali.
""")

# --- Tasvirni yuklash qismi ---
st.header("1. Tasvirni yuklash")
uploaded_file = st.file_uploader("Tasvirni yuklang...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # PIL kutubxonasi yordamida tasvirni ochish
    image = Image.open(uploaded_file)
    # Tasvirni OpenCV formatiga o'tkazish (NumPy massivi)
    # Rangli tasvir bo'lsa, uni RGBA'dan RGB'ga, keyin kulrang shkalaga o'tkazamiz
    img_np = np.array(image)
    
    if len(img_np.shape) == 3: # Agar tasvir rangli bo'lsa (RGB yoki RGBA)
        if img_np.shape[2] == 4: # RGBA bo'lsa, alfa kanalini olib tashlaymiz
            img_np = img_np[:, :, :3]
        original_img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else: # Agar tasvir allaqachon kulrang shkalada bo'lsa
        original_img_cv = img_np

    st.image(original_img_cv, caption='Yuklangan asl tasvir', use_column_width=True)

    # --- Mediana filtri sozlamalari ---
    st.header("2. Mediana filtri sozlamalari")
    
    # Kernel o'lchamini sozlash
    kernel_size = st.slider(
        "Mediana filtri kernel o'lchami (toq son)",
        min_value=3,
        max_value=15, # Katta o'lchamlar ham sinab ko'rish uchun
        value=5,
        step=2 # Faqat toq sonlar tanlash imkonini beradi
    )
    
    st.write(f"Tanlangan kernel o'lchami: **{kernel_size}x{kernel_size}**")

    # --- Qayta ishlash tugmasi ---
    st.header("3. Qayta ishlash")
    process_button = st.button("Tasvirni qayta ishlash va shovqin qo'shish")

    if process_button:
        with st.spinner('Tasvir qayta ishlanmoqda...'):
            # Tasvirga sun'iy "tuz va murch" shovqini qo'shish
            noise_added_img_cv = original_img_cv.copy() 
            row, col = noise_added_img_cv.shape
            s_vs_p = 0.5 
            amount = 0.02 

            # "Tuz" shovqini (oq piksellar)
            num_salt = np.ceil(amount * noise_added_img_cv.size * s_vs_p).astype(int)
            coords_salt = [np.random.randint(0, i - 1, num_salt) for i in noise_added_img_cv.shape]
            noise_added_img_cv[tuple(coords_salt)] = 255

            # "Murch" shovqini (qora piksellar)
            num_pepper = np.ceil(amount * noise_added_img_cv.size * (1.0 - s_vs_p)).astype(int)
            coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in noise_added_img_cv.shape]
            noise_added_img_cv[tuple(coords_pepper)] = 0
            
            st.success("Shovqin muvaffaqiyatli qo'shildi!")

            # Mediana filtrini qo'llash
            filtered_img_cv = cv2.medianBlur(noise_added_img_cv, kernel_size)
            st.success("Tasvir mediana filtri bilan tozalangan!")

            # --- Natijalarni ko'rsatish ---
            st.header("4. Natijalar")
            
            col1, col2 = st.columns(2) # Ikki ustun yaratish

            with col1:
                st.subheader("Shovqinli tasvir")
                st.image(noise_added_img_cv, caption="Sun'iy shovqin qo'shilgan", use_column_width=True)

            with col2:
                st.subheader("Filtrlangan tasvir")
                st.image(filtered_img_cv, caption=f"Mediana filtri ({kernel_size}x{kernel_size})", use_column_width=True)
            
            st.markdown("---")
            st.success("Jarayon yakunlandi!")

else:
    st.info("Boshlash uchun yuqoridagi 'Tasvirni yuklang...' tugmasi orqali tasvirni tanlang.")