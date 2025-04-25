# Proyek Analisis dan Prediksi Churn Pelanggan IndoHome

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan model prediksi churn pelanggan untuk layanan telekomunikasi IndoHome. Dengan menggunakan teknik machine learning dan analisis data, kami mengidentifikasi faktor-faktor yang mempengaruhi churn pelanggan dan membangun model prediktif untuk mengidentifikasi pelanggan yang berisiko berhenti berlangganan.

IndoHome, sebagai penyedia layanan fixed broadband, menghadapi tantangan dalam mempertahankan pelanggan di tengah persaingan industri telekomunikasi yang semakin ketat. Tingkat churn yang tinggi berdampak signifikan terhadap pendapatan dan pertumbuhan bisnis. Proyek ini dikembangkan untuk membantu tim pemasaran dan manajemen pelanggan dalam mengidentifikasi pelanggan berisiko tinggi dan mengimplementasikan strategi retensi yang efektif.

### Scope Proyek:
- Analisis faktor-faktor yang mempengaruhi churn pelanggan
- Pengembangan model prediksi churn dengan fokus pada recall tinggi
- Interpretasi model untuk insight bisnis
- Deployment model dalam bentuk aplikasi web interaktif
- Rekomendasi strategi retensi berbasis data

## Link Tableau Dashboard

### Dashboard 1 - Executive Overview
![Dashboard 1 - Executive Overview](Tableau%20Dashboard/Dashboard%201%20-%20Executive%20Overview.png)

Dashboard ini menyajikan ringkasan eksekutif tentang performa churn pelanggan, termasuk metrik utama, tren churn, dan KPI penting untuk pengambilan keputusan tingkat manajemen.

### Dashboard 2 - Customer Risk Segmentation
![Dashboard 2 - Customer Risk Segmentation](Tableau%20Dashboard/Dashboard%202%20-%20Customer%20Risk%20Segmentation.png)

Dashboard ini menampilkan segmentasi pelanggan berdasarkan tingkat risiko churn, memungkinkan tim pemasaran untuk memprioritaskan intervensi pada segmen pelanggan yang paling berisiko.

### Dashboard 3 - Feature Insights
![Dashboard 3 - Feature Insights](Tableau%20Dashboard/Dashboard%203%20-%20Feature%20Insights.png)

Dashboard ini menyajikan analisis mendalam tentang faktor-faktor yang mempengaruhi churn pelanggan, termasuk visualisasi kontribusi setiap fitur terhadap prediksi churn.

Dashboard Tableau lengkap dapat diakses melalui link berikut:
[IndoHome Churn Analysis Dashboard](https://public.tableau.com/views/IndoHomeChurnAnalysisDashboard/Dashboard1-ExecutiveOverview?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Link Halaman Streamlit

Aplikasi prediksi churn berbasis Streamlit dapat diakses melalui link berikut:
[Telco Customer Churn Prediction App](https://indohome-telco-churn-app-jaqqq4c38dcepxzzitrbzc.streamlit.app)

Aplikasi ini memungkinkan tim pemasaran untuk:
- Memprediksi churn pelanggan secara individu melalui form input
- Melakukan prediksi batch melalui upload file CSV
- Melihat interpretasi model dan faktor-faktor yang mempengaruhi prediksi
- Mendapatkan rekomendasi bisnis berdasarkan hasil prediksi

## Dokumentasi Penggunaan

### Cara Mengakses Dashboard Tableau
1. Klik link dashboard yang disediakan di atas
2. Gunakan filter interaktif untuk menyesuaikan tampilan data
3. Hover pada visualisasi untuk melihat detail tambahan

### Cara Menggunakan Aplikasi Streamlit
1. Akses link aplikasi Streamlit yang disediakan di atas
2. Pilih menu yang diinginkan dari sidebar (Home, Predict, Batch Prediction, dll)
3. Untuk prediksi individu:
   - Isi form dengan data pelanggan
   - Klik tombol "Predict" untuk melihat hasil
4. Untuk prediksi batch:
   - Upload file CSV dengan format yang sesuai
   - Klik tombol "Run Batch Prediction"
   - Download hasil prediksi dalam format CSV

### Menjalankan Kode Secara Lokal
1. Clone repositori:
   ```
   git clone https://github.com/username/Telco-Customer-Churn.git
   cd Telco-Customer-Churn
   ```
2. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan notebook Jupyter:
   ```
   jupyter notebook notebooks/
   ```
4. Jalankan aplikasi Streamlit lokal:
   ```
   cd deployment
   streamlit run app.py
   ```

## Metode yang Digunakan

### Analisis Data
- **Exploratory Data Analysis (EDA)**: Analisis univariat, bivariat, dan multivariat untuk memahami karakteristik data dan hubungan antar variabel
- **Feature Engineering**: Pembuatan fitur baru seperti TenureGroup, ServiceCount, dan PriceVariation untuk meningkatkan performa model
- **Data Preprocessing**: Penanganan missing values, encoding variabel kategorikal, dan normalisasi fitur numerik

### Pemodelan
- **Logistic Regression**: Model utama yang dipilih karena interpretabilitas tinggi dan performa yang baik
- **ADASYN (Adaptive Synthetic Sampling)**: Teknik resampling untuk mengatasi ketidakseimbangan kelas
- **Threshold Optimization**: Penyesuaian threshold klasifikasi untuk memaksimalkan F2-score
- **Pipeline Scikit-learn**: Implementasi pipeline preprocessing dan model untuk memastikan konsistensi

### Evaluasi Model
- **Cross-Validation**: 5-fold cross-validation untuk memastikan stabilitas model
- **Metrik Evaluasi**: F2-score, recall, precision, dan confusion matrix

### Visualisasi
- **Tableau**: Untuk dashboard interaktif dan visualisasi bisnis
- **Matplotlib & Seaborn**: Untuk visualisasi dalam analisis data
- **Plotly**: Untuk visualisasi interaktif dalam aplikasi Streamlit

## Instalasi dan Persyaratan

### Prasyarat
- Python 3.8+
- Jupyter Notebook
- Tableau Public (untuk melihat dashboard)
- Browser web modern

### Dependensi Python

streamlit
joblib
numpy
matplotlib
seaborn
scikit-learn
pandas
wheel
altair
plotly
imbalanced-learn


### Instalasi
1. Clone repositori:
   ```
   git clone https://github.com/username/Telco-Customer-Churn.git
   ```
2. Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):
   ```
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```
3. Instal dependensi:
   ```
   pip install -r requirements.txt
   ```

## Penjelasan Evaluasi dan Model

### Metrik Evaluasi
Kami menggunakan beberapa metrik untuk mengevaluasi performa model:

- **F2-score**: Metrik utama yang memberikan bobot lebih pada recall dibandingkan precision, sesuai dengan kebutuhan bisnis untuk meminimalkan false negative (pelanggan churn yang tidak terdeteksi)
- **Recall**: Mengukur kemampuan model untuk mendeteksi pelanggan yang benar-benar churn
- **Precision**: Mengukur akurasi model dalam memprediksi pelanggan yang churn
- **Confusion Matrix**: Memberikan gambaran lengkap tentang prediksi benar dan salah

### Hasil Evaluasi Model
- **F2-Score Test Set**: 0.7253
- **Recall Churn**: 0.83 (83% pelanggan churn berhasil terdeteksi)
- **Precision Churn**: 0.48 (48% prediksi churn adalah benar)
- **Threshold Optimal**: 0.43 (berdasarkan analisis F2-Score)

### Implementasi Model
Model Logistic Regression dengan ADASYN resampling dipilih sebagai model final karena:

1. **Interpretabilitas tinggi**: Koefisien model dapat langsung diinterpretasikan untuk insight bisnis
2. **Performa baik**: Mencapai recall tinggi (0.83) yang sesuai dengan kebutuhan bisnis
3. **Efisiensi komputasi**: Lebih ringan dibandingkan model ensemble, cocok untuk deployment
4. **Stabilitas**: Performa konsisten dalam cross-validation

Model diimplementasikan dalam pipeline scikit-learn yang mencakup:
- Preprocessing data (StandardScaler, OneHotEncoder)
- ADASYN resampling untuk mengatasi ketidakseimbangan kelas
- Logistic Regression dengan hyperparameter optimal

### Simulasi Dampak Finansial
Berdasarkan simulasi, implementasi model dapat menghemat:
- 16.6% biaya dibandingkan strategi promosi massal
- 62.3% kerugian dibandingkan tanpa strategi retensi

## Link dan Referensi

### Dataset
- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

### Referensi
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [Churn Prediction in Telecom Industry](https://ieeexplore.ieee.org/document/8258082)
- [Imbalanced-learn: ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)

## Catatan Tambahan

### Keterbatasan
- Dataset yang digunakan adalah dataset publik, bukan data asli IndoHome, sehingga mungkin tidak sepenuhnya mencerminkan karakteristik pelanggan IndoHome yang sebenarnya
- Model tidak mempertimbangkan faktor temporal (perubahan perilaku pelanggan dari waktu ke waktu)
- Tidak ada data tentang interaksi pelanggan dengan layanan pelanggan atau riwayat keluhan

### Saran Pengembangan
- Integrasi dengan sistem CRM untuk prediksi otomatis dan real-time
- Pengembangan model time-series untuk memprediksi churn dengan mempertimbangkan perubahan perilaku pelanggan
- Penambahan fitur dari sumber data lain seperti interaksi layanan pelanggan, riwayat keluhan, dan data penggunaan
- Implementasi A/B testing untuk strategi retensi berbasis model

### Rencana Monitoring dan Retraining
- Evaluasi performa model setiap bulan
- Retraining model setiap 3-6 bulan atau ketika terjadi perubahan signifikan dalam bisnis
- Monitoring distribusi fitur untuk mendeteksi data drift
- Pengumpulan feedback dari tim pemasaran untuk perbaikan model

---
