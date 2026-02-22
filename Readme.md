# Aksara Bali EEG Classification

Aplikasi berbasis **Deep Learning** untuk mengklasifikasikan sinyal EEG menjadi aksara Bali.

- **Backend** — FastAPI (Python 3.10) untuk preprocessing, pelatihan, dan prediksi.  
- **Frontend** — React + Vite untuk dashboard pengelolaan data, model, dan hasil.

---

## Fitur

- Preprocessing sinyal EEG (filtering, ekstraksi fitur, baseline reduction).  
- Pelatihan **LSTM** dengan cross-validation.  
- Manajemen model (simpan, muat, hapus).  
- Prediksi aksara Bali dari data EEG baru.  
- Visualisasi confusion matrix dan metrik pelatihan.

---

## Pengembangan Model

Bagian ini menjelaskan penggunaan skrip Python (`.py`) di folder ini untuk memproses data EEG dan melatih model LSTM.

### Struktur Direktori Data
Sistem ini menggunakan struktur folder standar `datasets/` untuk aliran data. Pastikan folder berikut tersedia:

- **Input `1D_Data.py`**:
  - `datasets/raw/CS_Train/` (Data EEG mentah `.mat`)
  - `datasets/baseline/` (Data baseline `.mat`)
- **Output `1D_Data.py` / Input `2D_Data.py`**:
  - `datasets/features/CS_Train/` (Fitur DE hasil ekstraksi)
- **Output `2D_Data.py` / Input `TrainLSTM.py`**:
  - `datasets/train/CS_Train/` (Data siap latih untuk LSTM)

### Cara Penggunaan

#### 1. Ekstraksi Fitur (`1D_Data.py`)
Membaca data dari `datasets/raw` dan menyimpan fitur ke `datasets/features`.
```bash
python 1D_Data.py
```

#### 2. Format Data LSTM (`2D_Data.py`)
Memproses fitur dari `datasets/features` dan menyimpannya ke `datasets/train`.
```bash
python 2D_Data.py
```

#### 3. Pelatihan Model (`TrainLSTM.py`)
Melatih model mengambil data dari `datasets/train`.
```bash
python TrainLSTM.py
```
Output meliputi akurasi pelatihan, validasi, dan confusion matrix.

## Struktur Repositori

```
.
├─ api/                      # Backend FastAPI
│  ├─ main.py
│  ├─ requirements.txt
│  └─ runtime.txt
├─ app/                      # Frontend React (Vite)
│  ├─ App.tsx
│  └─ package.json
├─ models/                   # Model terlatih (.h5)
├─ requirements.txt          # Kebutuhan Python global
```

> Direktori/berkas yang **tidak** boleh di-commit: `env_skripsi/`, `.venv/`, `__pycache__/`, `.env`, file model besar.

---

## Persyaratan

- Python **3.10.x**  
- Node.js **18.x** atau **20.x**  
- npm **9.x** atau **10.x**  
- pip **23.x** atau lebih baru  

> Untuk deployment production: Nginx + domain + sertifikat SSL (Let's Encrypt).

---

## Panduan Lengkap: Instalasi & Menjalankan Sistem

Panduan ini menjelaskan langkah demi langkah untuk menyiapkan dan menjalankan seluruh sistem dari awal.

### Langkah 1: Persiapkan Prasyarat

Pastikan sudah terinstal di komputer Anda:

| Software | Versi | Cek Versi |
|----------|-------|-----------|
| Python | 3.10.x | `python --version` |
| pip | 23.x+ | `pip --version` |
| Node.js | 18.x / 20.x | `node --version` |
| npm | 9.x / 10.x | `npm --version` |
| Git | Terbaru | `git --version` |

### Langkah 2: Clone Repositori

```bash
git clone <URL-repo-ini>
cd <nama-folder-repo>
```

### Langkah 3: Buat Virtual Environment Python

```bash
# Buat virtual environment
python -m venv env_skripsi

# Aktivasi (Windows PowerShell)
env_skripsi\Scripts\Activate.ps1

# Aktivasi (Windows CMD)
env_skripsi\Scripts\activate.bat

# Aktivasi (Linux/Mac)
source env_skripsi/bin/activate
```

> Setelah aktif, prompt terminal akan berubah menjadi `(env_skripsi)`.

### Langkah 4: Install Dependensi Python

```bash
# Dari root folder proyek (pastikan virtual env aktif)
pip install -r requirements.txt
```

Ini akan menginstal semua dependensi termasuk: `numpy`, `scipy`, `scikit-learn`, `tensorflow`, `keras`, `fastapi`, `uvicorn`, `matplotlib`, `pandas`, dll.

### Langkah 5: Siapkan Struktur Data

Pastikan folder dataset sudah tersedia dengan struktur berikut:

```
datasets/
├── raw/
│   └── CS_Train/              ← File .mat EEG mentah
├── baseline/                  ← File .mat baseline
├── features/
│   └── CS_Train/              ← (akan diisi otomatis oleh 1D_Data.py)
└── train/
    └── CS_Train/              ← (akan diisi otomatis oleh 2D_Data.py)
```

Buat folder yang belum ada:
```bash
mkdir -p datasets/raw/CS_Train
mkdir -p datasets/baseline
mkdir -p datasets/features/CS_Train
mkdir -p datasets/train/CS_Train
```

> **Windows CMD:** Gunakan `mkdir datasets\raw\CS_Train` dst.

Letakkan file `.mat` Anda di folder yang sesuai.

### Langkah 6: Jalankan Pipeline Data Processing

Pipeline dijalankan **secara berurutan** — output dari satu skrip menjadi input skrip berikutnya.

#### 6a. Ekstraksi Fitur — `1D_Data.py`

Mengubah data EEG mentah menjadi fitur Differential Entropy (DE).

```bash
# Dengan baseline reduction (default, direkomendasikan)
python 1D_Data.py with

# Tanpa baseline reduction
python 1D_Data.py without
```

- **Input:** `datasets/raw/CS_Train/*.mat` + `datasets/baseline/*.mat`
- **Output:** `datasets/features/CS_Train/DE_*.mat`

#### 6b. Format Data LSTM — `2D_Data.py`

Mengubah fitur menjadi format 2D yang siap untuk pelatihan LSTM.

```bash
python 2D_Data.py
```

- **Input:** `datasets/features/CS_Train/*.mat`
- **Output:** `datasets/train/CS_Train/*.mat`

#### 6c. Pelatihan Model — `TrainLSTM.py`

Melatih model LSTM dengan 10-fold cross-validation.

```bash
python TrainLSTM.py
```

- **Input:** `datasets/train/CS_Train/*.mat`
- **Output:** Hasil akurasi, loss, confusion matrix, dan metrik per partisipan.

### Langkah 7: Jalankan Backend (FastAPI)

```bash
cd api
pip install -r requirements.txt    # Install dependensi tambahan API (jika ada)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend tersedia di:
- API: `http://localhost:8000`
- Swagger UI (dokumentasi API): `http://localhost:8000/docs`

> Flag `--reload` memungkinkan auto-restart saat kode berubah. Hapus untuk mode production.

### Langkah 8: Jalankan Frontend (React + Vite)

Buka **terminal baru** (jangan tutup terminal backend):

```bash
cd app
npm install             # Install dependensi Node.js
npm run dev             # Jalankan dev server
```

> Pastikan file `.env` di `app/` sudah berisi `VITE_API_URL=http://localhost:8000` sebelum menjalankan.

Frontend tersedia di: `http://localhost:5173`

### Rangkuman Urutan Menjalankan

```
1. Aktivasi virtual environment
2. python 1D_Data.py with        ← Ekstraksi fitur
3. python 2D_Data.py             ← Format data
4. python TrainLSTM.py           ← Latih model
5. cd api && uvicorn main:app    ← Backend
6. cd app && npm run dev         ← Frontend (terminal baru)
```

---

## Konfigurasi

### 1) Backend (`api/.env` atau variabel lingkungan)

Tambahkan variabel yang diperlukan backend (contoh):

```env
MODEL_DIR=./models
```

> Tambahkan variabel lain sesuai kebutuhan kode Anda.

Jika mengaktifkan CORS, pastikan origin frontend production:

```python
# di api/main.py (contoh)
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://app.domainmu.com",  # ganti dengan domain frontend Anda
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2) Frontend (`app/.env` atau `.env.production`)

```env
# URL API backend
VITE_API_URL=https://api.domainmu.com     # ganti sesuai domain backend Anda
```

Saat pengembangan lokal:

```env
VITE_API_URL=http://localhost:8000
```

---

## Instalasi & Menjalankan (Untuk Pengguna GitHub)

### 1) Clone repositori

```bash
git clone <URL-repo-ini>
cd <nama-folder-repo>
```

### 2) Backend (FastAPI)

```bash
cd api
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Backend tersedia di: `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### 3) Frontend (React + Vite)

Buka terminal baru:

```bash
cd app
npm install
# pastikan VITE_API_URL sudah benar (lihat bagian Konfigurasi)
npm run dev
```

Frontend tersedia di: `http://localhost:5173`

---

## Build & Artefak

### Build Frontend untuk Production

```bash
cd app
npm ci
npm run build      # output: app/dist
```

### Menjalankan Backend Tanpa `--reload` (simulasi production lokal)

```bash
cd api
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## Endpoint

- Frontend (Dashboard): `https://app.domainmu.com`  
- Backend API: `https://api.domainmu.com`  
- Swagger: `https://api.domainmu.com/docs`  

Untuk lokal:

- Frontend: `http://localhost:5173`  
- Backend: `http://localhost:8000`  
- Swagger: `http://localhost:8000/docs`

---

## Cara Menggunakan

1. **Preprocessing Data EEG**  
   - Buka dashboard.  
   - Upload file baseline dan data **`.mat`**.  
   - Jalankan preprocessing untuk ekstraksi fitur.

2. **Pelatihan Model**  
   - Setelah preprocessing, auto menjalankan training **LSTM**.  
   - Model disimpan ke folder `models/` (atau sesuai konfigurasi `MODEL_DIR`).

3. **Prediksi**  
   - Pilih model terlatih di sidebar.  
   - Upload data EEG baru untuk inferensi.

---

## Catatan Penting

- File model (`.h5`) dan data (`.mat`) sebaiknya **ringkas** agar proses stabil.  
- **Jangan commit**: `env_skripsi/`, `.venv/`, `__pycache__/`, `.env`, model besar.  
- Pastikan lingkungan server/VPS sesuai versi pada bagian **Persyaratan**.


---

## Kontribusi

Buka **issue** untuk diskusi/bug, atau ajukan **pull request** untuk perbaikan/fitur.

---

## Lisensi

**MIT License**
© I Dewa Gede Mahesta Parawangsa  
[https://www.linkedin.com/in/demahesta](https://www.linkedin.com/in/demahesta)