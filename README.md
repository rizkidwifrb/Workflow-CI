# 🚀 Workflow CI - Automated Machine Learning Pipeline

Repository ini berisi implementasi **Continuous Integration (CI)** untuk proses training model Machine Learning menggunakan **MLflow Project** dan **GitHub Actions**.

Pipeline ini memungkinkan proses training model berjalan **secara otomatis setiap kali terjadi perubahan kode** pada repository.

---

## 📌 Tujuan Project

Membangun workflow otomatis untuk:

* Menjalankan training model Machine Learning
* Melacak eksperimen menggunakan MLflow
* Mengotomatisasi proses retraining melalui GitHub Actions
* Memastikan pipeline berjalan konsisten pada environment CI

---

## ⚙️ Workflow Overview

Pipeline CI berjalan dengan alur berikut:

1. Developer melakukan **push** ke repository
2. GitHub Actions otomatis dijalankan
3. MLflow Project dieksekusi
4. Script modelling dijalankan
5. Model dan artifact tersimpan sebagai experiment MLflow

```
Push Code
   ↓
GitHub Actions
   ↓
MLflow Run
   ↓
Model Training
   ↓
Artifacts & Metrics Logged
```

---

## 🗂️ Struktur Repository

```
Workflow-CI
│
├── .github/workflows/
│   └── ci.yml
│
└── MLProject
    ├── MLproject
    ├── modelling.py
    └── student_performance_processed.csv
```

---

## 🧠 Teknologi yang Digunakan

* Python 3.10
* MLflow
* Pandas
* Scikit-learn
* GitHub Actions

---

## ▶️ Menjalankan Project Secara Lokal

Pastikan dependency telah terinstall:

```bash
pip install -r requirements.txt
```

Jalankan MLflow Project:

```bash
mlflow run MLProject --env-manager=local
```

---

## 🔄 Continuous Integration (CI)

Workflow CI akan berjalan otomatis ketika:

* Push ke branch `main`
* Update source code modelling
* Perubahan pipeline

GitHub Actions akan:

✅ Setup Python environment
✅ Install dependencies
✅ Menjalankan MLflow Project
✅ Melakukan training model otomatis

---

## 📊 Experiment Tracking

Semua hasil training dicatat menggunakan MLflow, meliputi:

* Parameter model
* Metrics evaluasi
* Model artifacts
* Training logs

---

## ✅ Status Workflow

Pipeline berhasil dijalankan apabila GitHub Actions menunjukkan status:

```
✔ SUCCESS
```

---

## 📸 Bukti Eksekusi

Tambahkan screenshot berikut pada repository:

* GitHub Actions successful run
* MLflow experiment dashboard
* MLflow artifacts

---

## 👨‍💻 Author

Machine Learning Workflow CI Implementation
for System Machine Learning Submission
