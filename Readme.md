# Aksara Bali EEG Classification

## Deskripsi Proyek

Aksara Bali EEG Classification adalah aplikasi berbasis **Deep Learning** untuk mengklasifikasikan sinyal EEG menjadi aksara Bali.

- **Backend**: API berbasis FastAPI (Python 3.10) untuk preprocessing, pelatihan, dan prediksi model.
- **Frontend**: Dashboard interaktif berbasis React + Vite untuk mengelola data, model, dan hasil klasifikasi.

## Fitur Utama

- Preprocessing sinyal EEG (filtering, ekstraksi fitur, baseline reduction)
- Pelatihan model **Long Short-Term Memory (LSTM)** dengan cross-validation
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
├─ app/                      # Frontend React (Vite)
│  ├─ App.tsx
│  └─ package.json
├─ .env
├─ requirements.txt          # Kebutuhan Python global
(jangan di-push ke repo)

````

---

## Persyaratan Sistem

- **Python**: 3.10.x  
- **Node.js**: 18.x atau 20.x  
- **npm**: 9.x atau 10.x  
- **pip**: 23.x atau lebih baru  
- **Nginx** (untuk reverse proxy & serving frontend)  
- **Domain** (misal: `api.domainmu.com` untuk backend dan `app.domainmu.com` untuk frontend) + **SSL** (Let's Encrypt)

> **Catatan:** Di production, **jangan** menjalankan Vite `npm run dev` atau `uvicorn --reload`. Gunakan **build statis** untuk frontend dan **service** untuk backend di belakang Nginx (HTTPS).

---

## Konfigurasi Production (Disarankan)

### 1) Backend (FastAPI) sebagai service

1. **Siapkan virtual env & dependensi**
   ```bash
   cd api
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
````

2. **(Opsional) Konfigurasi CORS di `api/main.py`**
   Pastikan origin frontend production diizinkan:

   ```python
   from fastapi.middleware.cors import CORSMiddleware

   origins = [
       "http://localhost:5173",    # <— GANTI: domain frontend production
   ]

   app.add_middleware(
       CORSMiddleware,
       allow_origins=origins,
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. **Jalankan sebagai service (systemd)**
   Buat file `/etc/systemd/system/aksarabali-api.service`:

   ```ini
   [Unit]
   Description=AksaraBali FastAPI
   After=network.target

   [Service]
   User=www-data
   Group=www-data
   WorkingDirectory=/path/to/repo/api
   Environment="PATH=/path/to/repo/api/.venv/bin"
   ExecStart=/path/to/repo/api/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
   # Jika ingin port internal 9000, ganti --port 9000 (sesuaikan Nginx di bawah)
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Aktifkan:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable aksarabali-api
   sudo systemctl start aksarabali-api
   sudo systemctl status aksarabali-api
   ```

### 2) Frontend (React + Vite) build statis

1. **Build**

   ```bash
   cd app
   npm ci        # atau npm install
   npm run build # menghasilkan folder dist/
   ```

2. **Deploy hasil build** (serve via Nginx)

   ```bash
   sudo mkdir -p /var/www/aksarabali-app
   sudo rsync -a dist/ /var/www/aksarabali-app/
   ```

### 3) Nginx (reverse proxy + static hosting)

**Site untuk Backend** (`/etc/nginx/sites-available/api.domainmu.com`)

```nginx
server {
    server_name api.domainmu.com;

    location / {
        proxy_pass http://127.0.0.1:8000;  # <— sesuaikan jika service pakai port lain
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Site untuk Frontend** (`/etc/nginx/sites-available/app.domainmu.com`)

```nginx
server {
    server_name app.domainmu.com;

    root /var/www/aksarabali-app;
    index index.html;

    # Vite SPA fallback
    location / {
        try_files $uri /index.html;
    }
}
```

Aktifkan site & reload:

```bash
sudo ln -s /etc/nginx/sites-available/api.domainmu.com /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/app.domainmu.com /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Tambahkan SSL (Let's Encrypt)**

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.domainmu.com
sudo certbot --nginx -d app.domainmu.com
```

---

## Konfigurasi Environment (Production)

### Frontend (`app/.env` atau `.env.production`)

```env
# GANTI dari http://localhost:8000 -> URL API production
VITE_API_URL=https://api.domainmu.com
```

> Setelah mengubah `.env`, lakukan `npm run build` ulang.

### Backend (opsional `.env` untuk setting lain)

Jika ada konfigurasi di kode yang membaca environment (misal path model, secret, dsb.), definisikan di `.env` lalu baca via `os.getenv(...)`.

---

## Endpoint Production

* **Frontend (Dashboard):** `https://app.domainmu.com`
* **Backend API:** `https://api.domainmu.com`
* **Dokumentasi API (Swagger):** `https://api.domainmu.com/docs`

> Di dokumentasi/README, **hindari** mencantumkan `http://localhost:5173` atau `http://localhost:8000` untuk pengguna akhir. Gunakan domain production di atas.

---

## Cara Penggunaan (Production)

1. **Preprocessing Data EEG**

   * Masuk ke dashboard: `https://app.domainmu.com`
   * Upload file baseline & data EEG pelatihan (`.mat`).
   * Jalankan preprocessing untuk ekstraksi fitur.

2. **Pelatihan Model**

   * Setelah preprocessing, mulai training model LSTM.
   * Model otomatis disimpan (default: `api/models/` di server).

3. **Prediksi**

   * Pilih model terlatih.
   * Upload data EEG baru untuk prediksi aksara Bali.

4. **Visualisasi**

   * Lihat **confusion matrix** dan metrik pelatihan di dashboard.

---

## Checklist: Ganti Link `localhost` ke Production

* **Frontend**: `app/.env`

  ```diff
  - VITE_API_URL=http://localhost:8000
  + VITE_API_URL=https://api.domainmu.com
  ```
* **Backend CORS** (`api/main.py`)

  ```diff
  - origins = ["http://localhost:5173"]
  + origins = ["https://app.domainmu.com"]
  ```
* **Dokumentasi/README**: gunakan

  * `https://app.domainmu.com` (Frontend)
  * `https://api.domainmu.com` (Backend, `/docs`)

> Jika **port internal backend** diubah ke **9000**, sesuaikan:
> `ExecStart` service systemd (`--port 9000`) dan `proxy_pass` Nginx (`http://127.0.0.1:9000`).
> URL publik **tetap** di 443/HTTPS melalui Nginx.

---

## Mode Pengembangan (opsional, lokal)

* Backend: `uvicorn main:app --reload --port 8000`
* Frontend: `npm run dev` (Vite di `http://localhost:5173`, pastikan `VITE_API_URL=http://localhost:8000`)

> Hanya untuk development lokal. Production mengikuti “Konfigurasi Production”.

---

## Catatan Penting

* Pastikan file model (`.h5`) dan data (`.mat`) **tidak terlalu besar** agar proses berjalan lancar.
* **Jangan push** folder `env_skripsi/`, `__pycache__/`, dan file `.env` ke repository.
* Untuk deployment di **server/VPS**, pastikan environment sesuai versi pada bagian *Persyaratan Sistem*.

> Rekomendasi `.gitignore` singkat:
>
> ```
> env_skripsi/
> .venv/
> __pycache__/
> *.pyc
> .env
> models/*.h5
> dist/
> node_modules/
> ```

---

## Kontribusi

Silakan buat **pull request** atau **issue** jika ingin berkontribusi atau menemukan bug.

---

## Lisensi

**MIT License** (tambahkan file `LICENSE` jika belum ada)
© I Dewa Gede Mahesta Parawangsa
[www.linkedin.com/in/demahesta](https://www.linkedin.com/in/demahesta)