# Aksara Bali EEG Classification

## Deskripsi Proyek

Aksara Bali EEG Classification adalah aplikasi berbasis Machine Learning untuk mengklasifikasikan sinyal EEG menjadi aksara Bali.

- **Backend**: API berbasis FastAPI (Python 3.10) untuk preprocessing, pelatihan, dan prediksi model.
- **Frontend**: Dashboard interaktif berbasis React + Vite untuk mengelola data, model, dan hasil klasifikasi.

## Fitur Utama

- Preprocessing sinyal EEG (filtering, ekstraksi fitur, baseline reduction)
- Pelatihan model LSTM dengan cross-validation
- Manajemen model (simpan, muat, hapus)
- Prediksi aksara Bali dari data EEG baru
- Visualisasi confusion matrix dan metrik pelatihan

---

## Struktur Direktori

```

.
├─ api/                      # Backend FastAPI
│  ├─ main.py
│  ├─ requirements.txt
│  └─ runtime.txt
├─ models/
├─ app/                      # Frontend React
│  ├─ App.tsx
│  └─ package.json
├─ .env
├─ components/
├─ requirements.txt          # Kebutuhan Python global
├─ pytest.ini                # Konfigurasi testing
├─ Readme.md                 # Dokumentasi proyek
└─ env_skripsi/              # Virtual environment (jangan di-push ke repo)

````

## Persyaratan Sistem

- **Python**: 3.10.
- **Node.js**: 18.x atau 20.x (untuk frontend)
- **npm**: 9.x atau 10.x
- **pip**: 23.x atau lebih baru

---

## Cara Setup & Menjalankan Proyek (Tanpa Docker)

### 1. Setup Backend (FastAPI)

#### a. Buat Virtual Environment

```bash
python -m venv env_skripsi
# Aktifkan:
# Linux/Mac
source env_skripsi/bin/activate
# Windows
env_skripsi\Scripts\activate
````

#### b. Install Dependencies

```bash
pip install -r requirements.txt
```

#### c. Jalankan API FastAPI

```bash
cd api
uvicorn main:app --reload --port 8000
```

API akan berjalan di [http://localhost:8000](http://localhost:8000). Dokumentasi API tersedia di [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Setup Frontend (React + Vite)

#### a. Install Dependencies

```bash
cd app
npm install
```

#### b. Konfigurasi Environment

Buat/ubah file `.env` di folder `app`:

```env
VITE_API_URL=http://localhost:8000
```

#### c. Jalankan Frontend

```bash
npm run dev
```

Frontend akan berjalan di [http://localhost:5173](http://localhost:5173).

---

## Cara Penggunaan

1. **Preprocessing Data EEG**

   * Upload file baseline dan data EEG pelatihan (`.mat`) melalui dashboard.
   * Jalankan preprocessing untuk ekstraksi fitur.
2. **Pelatihan Model**

   * Setelah preprocessing, mulai pelatihan model LSTM.
   * Model akan disimpan di folder `api/models/`.
3. **Prediksi**

   * Pilih model yang sudah dilatih.
   * Upload data EEG baru untuk prediksi aksara Bali.
4. **Visualisasi**

   * Lihat **confusion matrix** dan metrik pelatihan pada dashboard.