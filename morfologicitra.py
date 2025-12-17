import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI UTAMA ---
FILENAME = 'Snail.jpg'  # Ganti dengan citra input Anda yang bersih
# Daftar 6 eksperimen sesuai permintaan modul Anda
EXPERIMENTS = [
    {"op": "Erosi", "strel": "LINE", "R": 10},
    {"op": "Erosi", "strel": "DISK", "R": 5},
    {"op": "Dilasi", "strel": "DIAMOND", "R": 8},
    {"op": "Dilasi", "strel": "SQUARE", "R": 10},
    {"op": "Opening", "strel": "DIAMOND", "R": 4},
    {"op": "Closing", "strel": "LINE", "R": 10},
]
# -------------------------

def get_kernel(R, strel_shape_name):
    """Membuat Structuring Element (Strel) berdasarkan bentuk dan radius R."""
    size = 2 * R + 1
    
    if strel_shape_name in ['RECT', 'SQUARE']:
        # SQUARE
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif strel_shape_name in ['DISK', 'ELLIPSE']:
        # DISK (Analog dengan ELLIPSE)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif strel_shape_name in ['CROSS', 'DIAMOND']:
        # DIAMOND (Analog dengan CROSS)
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    elif strel_shape_name == 'LINE':
        # LINE (Disimulasikan sebagai garis vertikal tipis, lebar 1 piksel)
        # Strel: 1 piksel lebar, 2*R+1 tinggi
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    
    # Default jika tidak ada yang cocok
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def operasi_morfologi_spesifik(filename, experiments_list):
    
    # 1. MEMBACA & MEMPROSES INPUT
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat file '{filename}'. Pastikan file ada.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Konversi ke Citra Biner (Input Morfologi)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 2. MENJALANKAN 6 EKSPERIMEN
    results = []
    
    for i, exp in enumerate(experiments_list):
        
        # Buat kernel yang spesifik untuk eksperimen ini
        kernel = get_kernel(exp['R'], exp['strel'])
        
        # Tentukan operasi
        if exp['op'] == 'Erosi':
            result_img = cv2.erode(img_binary, kernel, iterations=1)
        elif exp['op'] == 'Dilasi':
            result_img = cv2.dilate(img_binary, kernel, iterations=1)
        elif exp['op'] == 'Opening':
            result_img = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        elif exp['op'] == 'Closing':
            result_img = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        else:
            result_img = np.zeros_like(img_binary)
        
        # Simpan hasil untuk plot
        title = f"{i+3}. {exp['op']} ({exp['strel']}, R={exp['R']})"
        results.append({'img': result_img, 'title': title})

    # 3. MENAMPILKAN HASIL (8 Plot: 1 Asli, 1 Biner, 6 Hasil)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Praktek 7: Operasi Morfologi - 6 Eksperimen", fontsize=16)
    axes_flat = axes.flat
    
    # Plot 1: Citra Asli
    axes_flat[0].imshow(img_rgb)
    axes_flat[0].set_title("1. Citra Asli (RGB)")
    axes_flat[0].axis('off')
    
    # Plot 2: Citra Biner Input
    axes_flat[1].imshow(img_binary, cmap='gray')
    axes_flat[1].set_title("2. Citra Biner Input")
    axes_flat[1].axis('off')
    
    # Plot 3 (Dibuat kosong agar plot 3.1 s.d 3.6 terlihat lebih baik)
    axes_flat[2].set_visible(False)
    
    # Plot 4 s.d 9: Hasil 6 Eksperimen
    for i in range(len(results)):
        axes_flat[i+3].imshow(results[i]['img'], cmap='gray')
        axes_flat[i+3].set_title(results[i]['title'])
        axes_flat[i+3].axis('off')
        
    plt.subplots_adjust(hspace=0.3)
    plt.show(block=True) 

# --- JALANKAN PROGRAM PRAKTEK 7 ---
print(f"Memulai Operasi Morfologi pada '{FILENAME}' dengan 6 konfigurasi spesifik.")
operasi_morfologi_spesifik(FILENAME, EXPERIMENTS)