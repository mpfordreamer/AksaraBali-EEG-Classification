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