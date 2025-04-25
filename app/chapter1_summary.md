
<div style="text-align: center;">
  <h1 style="background-color: #E60000; color: white; padding: 10px 20px; margin-bottom: 20px; border-radius: 10px; font-weight: bold;">
  Chapter 1: Business Problem Understanding
  </h1>
</div>

---

### 1.1 | Context

Perusahaan telekomunikasi **IndoHome** adalah salah satu penyedia layanan internet tetap (fixed broadband) terbesar di Indonesia. Pada kuartal terakhir tahun 2021, IndoHome mencatat lonjakan churn sebesar **29,11%**. Ini menjadi sinyal bahaya karena kehilangan pelanggan tidak hanya menurunkan pendapatan tetapi juga meningkatkan biaya operasional.

Biaya memperoleh pelanggan baru (*Customer Acquisition Cost*) bisa 5x lebih tinggi dari biaya mempertahankan pelanggan (*Customer Retention Cost*). Pendekatan berbasis data seperti *machine learning* memungkinkan IndoHome mengidentifikasi pelanggan berisiko churn dan memberikan intervensi personal.

---

### 1.2 | Problem Statement

CMO IndoHome menghadapi:
- Promosi ke pelanggan yang sebenarnya loyal (operation loss)
- Kehilangan pelanggan berharga (profit loss)
- Keputusan tidak tepat akibat kurang insight

**Solusi:**
- Sistem prediktif berbasis machine learning
- Identifikasi fitur utama penyebab churn
- Rekomendasi intervensi yang efisien dan personal

---

### 1.3 | Business Goals

Tujuan proyek ini adalah:
1. Membangun model prediktif churn berbasis perilaku pelanggan
2. Menghasilkan actionable insight untuk tim marketing
3. Memberikan dasar pengambilan keputusan promosi & loyalti
4. Mengurangi kerugian finansial akibat churn

---

### 1.4 | Stakeholders

**Chief Marketing Officer (CMO)** sebagai pengguna utama model:
- Menargetkan pelanggan berisiko
- Meningkatkan efisiensi promosi
- Merancang program loyalitas berbasis prediksi

---

### 1.5 | Metric Evaluation

**Tanpa model:**

| Skenario                  | Biaya Promosi      | Kehilangan Profit     | Total Kerugian         |
|---------------------------|---------------------|-------------------------|--------------------------|
| Semua diberi promosi     | Rp2.112.900.000     | Rp0                     | Rp2.112.900.000          |
| Tidak ada promosi         | Rp0                 | Rp5.125.000.000         | Rp5.125.000.000          |

**Dengan model:**

- Model meminimalkan False Negative (FN)
- **Metrik utama:** Recall dan F2-Score
- **Metrik pendamping:** Precision dan PRC-AUC
