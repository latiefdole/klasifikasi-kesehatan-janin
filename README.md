# Laporan Proyek Machine Learning - Abdul Latif

## Daftar Isi
1. [Domain Proyek](#domain-proyek)
2. [Business Understanding](#business-understanding)
   - [Problem Statements](#problem-statements)
   - [Goals](#goals)
   - [Solution Statements](#solution-statements)
3. [Data Understanding](#data-understanding)
   - [Deskripsi Variabel](#variabel-variabel)
4. [Data Visualization](#data-visualization)
5. [Data Preparation](#data-preparation)
6. [Modeling](#modeling)
7. [Evaluation](#evaluation)
8. [Conclusion](#conclusion)

## Domain Proyek

Penurunan angka kematian anakmerupakan salah satu aspek utama dari Tujuan Pembangunan Berkelanjutan (Sustainable Development Goals, SDGs) yang ditetapkan oleh Perserikatan Bangsa-Bangsa (PBB). Indikator ini juga menjadi parameter penting dalam menilai kemajuan suatu negara dalam bidang kesehatan masyarakat. PBB menargetkan agar pada tahun 2030, setiap negara mampu mengakhiri kematian anak yang dapat dicegah, khususnya pada bayi baru lahir dan anak di bawah usia 5 tahun. Target global ini menetapkan angka kematian balita di bawah 25 per 1.000 kelahiran hidup.

Sebagai isu yang berkaitan erat, kematian ibu juga menjadi perhatian global. Data tahun 2017 mencatat sekitar 295.000 kematian ibu yang terjadi selama kehamilan dan persalinan. Sebagian besar kematian ini (94%) terjadi di negara-negara dengan sumber daya terbatas, dan mayoritas kasus tersebut dapat dicegah melalui intervensi medis yang tepat.

Dalam konteks ini, penggunaan Cardiotocogram (CTG) menjadi alat diagnostik yang esensial dalam memantau kesehatan janin. CTG memungkinkan para profesional kesehatan untuk menilai detak jantung janin (Fetal Heart Rate, FHR), gerakan janin, serta kontraksi uterus secara non-invasif. Penggunaan CTG menawarkan opsi yang relatif terjangkau dan mudah diakses untuk membantu menurunkan angka kematian ibu dan anak. Alat ini bekerja dengan mengirimkan pulsa ultrasonik dan kemudian membaca responsnya, yang memberikan data penting tentang status fisiologis janin selama kehamilan.
  
Referensi: [Sisporto 2.0: A program for automated analysis of cardiotocograms
](https://onlinelibrary.wiley.com/doi/10.1002/1520-6661(200009/10)9:5%3C311::AID-MFM12%3E3.0.CO;2-9)

## Business Understanding

### Problem Statements

- Bagaimana data dari hasil CTG dapat membantu mengklasifikasikan kesehatan janin?
- Bagaimana model machine learning dapat digunakan untuk memprediksi hasil klasifikasi CTG secara otomatis?

### Goals

- Membangun model machine learning yang dapat mengklasifikasikan kondisi janin berdasarkan data CTG.
- Meningkatkan akurasi prediksi klasifikasi kesehatan janin menjadi lebih baik dari baseline model.

### Solution Statements

- Menggunakan beberapa algoritma machine learning (seperti Random Forest, SVM) untuk memprediksi klasifikasi kesehatan janin. Analisis lebih lanjut difokuskan pada pemetaan hubungan antara parameter CTG dengan klasifikasi klinis yang dilakukan oleh para ahli obstetri.
- Mengoptimalkan model melalui hyperparameter tuning untuk meningkatkan performa model yang dipilih.

## Data Understanding

Dataset ini terdiri dari 2.126 rekaman hasil pemeriksaan Cardiotocogram yang telah diklasifikasikan oleh ahli menjadi tiga kategori yaitu:
- Normal: Kategori ini mencakup hasil CTG yang menunjukkan aktivitas janin yang sehat tanpa adanya indikasi masalah.
- Mencurigakan (Suspect): Kategori ini mencakup hasil CTG yang menunjukkan beberapa anomali yang mungkin memerlukan pemeriksaan lebih lanjut.
- Patologis: Kategori ini mencakup hasil yang menunjukkan adanya masalah serius pada janin yang membutuhkan intervensi medis segera.

Data yang dikumpulkan mencakup berbagai parameter dari hasil CTG, seperti detak jantung janin, variabilitas detak jantung, percepatan dan deselerasi detak jantung, serta pola kontraksi rahim. Analisis dilakukan dengan menggunakan teknik statistik dan pembelajaran mesin untuk memprediksi klasifikasi kondisi janin berdasarkan data tersebut.
Data dapat diakses dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Cardiotocography).

### Variabel-variabel

- **Fetal Heart Rate (FHR)**: Baseline Detak jantung janin
- **Accelerations**: Akselerasi janin per detik
- **Fetal_Movement**: Gerakan janin
- **Uterine_Constraction**: Jumlah kontraksi uterus per detik
- dst.

## Data Visualization

Visualisasi data atau Exploratory Data Analysis (EDA) untuk memahami distribusi dan pola data.

![alt text](image.png)

## Data Preparation

Proses data preparation meliputi:
- Mengatasi missing values
- Normalisasi atau standardisasi data
- Pembagian data menjadi data latih dan data uji

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan setiap tahapan dan alasan mengapa dilakukan.

## Modeling

Pada bagian ini, dilakukan proses pemodelan dengan menggunakan algoritma machine learning. Model yang diusulkan meliputi algoritma seperti **Random Forest**, **Support Vector Machine (SVM)**, dan **k-Nearest Neighbors (k-NN)**. Parameter yang digunakan pada proses pemodelan dijelaskan, termasuk hyperparameter tuning yang dilakukan untuk meningkatkan performa model.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan hyperparameter tuning, jelaskan prosesnya dan bagaimana hal tersebut meningkatkan performa model.

## Evaluation

Pada bagian ini, metrik evaluasi yang digunakan adalah **akurasi**, **precision**, **recall**, dan **F1 score**. Penilaian dilakukan berdasarkan kemampuan model dalam mengklasifikasikan data CTG ke dalam kategori Normal, Suspect, atau Patologis.

Hasil evaluasi menunjukkan bahwa algoritma **Random Forest** memberikan hasil terbaik dengan akurasi **X%**, precision **Y%**, dan F1 score **Z%**.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menjelaskan formula metrik evaluasi dan bagaimana metrik tersebut bekerja.
- Menyertakan perbandingan hasil dari beberapa model untuk menunjukkan model terbaik.

## Conclusion

Studi ini memiliki implikasi penting dalam upaya menurunkan angka kematian ibu dan anak, terutama di negara-negara dengan sumber daya terbatas. Dengan pemanfaatan teknologi yang terjangkau seperti CTG, diharapkan para tenaga medis dapat lebih cepat dan akurat dalam mendeteksi komplikasi kehamilan, sehingga intervensi medis dapat dilakukan lebih awal. Hasil penelitian ini juga berpotensi untuk memberikan kontribusi dalam pengembangan protokol klinis yang lebih baik dalam pemantauan kehamilan.

Cardiotocogram menyediakan informasi yang penting dan mudah diakses mengenai kesehatan janin, yang dapat membantu mengurangi angka kematian ibu dan anak. Dengan pemanfaatan teknologi ini, serta dukungan dari analisis berbasis data, diharapkan proses deteksi dini masalah kesehatan janin dapat lebih ditingkatkan



