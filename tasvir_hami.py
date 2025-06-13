import cv2

# Kulrang tasvirni o'qish (yoki o'zingizning Brayl tasviringiz)
img_gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if img_gray is not None:
    # .shape atributi (balandlik, kenglik) yoki (balandlik, kenglik, kanallar_soni) qaytaradi
    height_gray, width_gray = img_gray.shape
    print(f"Kulrang tasvir o'lchamlari: {width_gray}x{height_gray}")
    # Tasvir hajmi (baytlarda)
    # img_gray.itemsize har bir pikselning baytdagi hajmini beradi (8-bitli tasvir uchun 1 bayt)
    size_gray = img_gray.size * img_gray.itemsize
    print(f"Kulrang tasvirning xotira hajmi: {size_gray} bayt")
else:
    print("Xato: Kulrang tasvir topilmadi.")

print("-" * 30)

# Rangli tasvirni o'qish (agar sinash uchun rangli tasviringiz bo'lsa)
# Masalan, 'image copy 2.jpg'
img_color = cv2.imread('image copy 2.png', cv2.IMREAD_COLOR)

if img_color is not None:
    height_color, width_color, channels_color = img_color.shape
    print(f"Rangli tasvir o'lchamlari: {width_color}x{height_color}, Kanallar: {channels_color}")
    size_color = img_color.size * img_color.itemsize
    print(f"Rangli tasvirning xotira hajmi: {size_color} bayt")
else:
    print("Xato: Rangli tasvir topilmadi (color_image.jpg mavjudligini tekshiring).")