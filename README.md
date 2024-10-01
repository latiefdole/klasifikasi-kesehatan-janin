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

![alt text](https://raw.githubusercontent.com/latiefdole/klasifikasi-kesehatan-janin/refs/heads/main/eda.png)
![alt text](https://raw.githubusercontent.com/latiefdole/klasifikasi-kesehatan-janin/refs/heads/main/eda2.png)

## Data Preparation

Proses data preparation meliputi:
- Mengatasi missing values
- Normalisasi atau standardisasi data
- Pembagian data menjadi data latih dan data uji

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan setiap tahapan dan alasan mengapa dilakukan.

## Modeling

Dalam tahapan modeling ini, kami melakukan eksperimen dengan beberapa algoritma klasifikasi untuk mengevaluasi kinerja mereka dalam memprediksi target variabel. Dua algoritma yang digunakan adalah **Decision Tree** dan **Random Forest**, yang diterapkan pada dataset dengan teknik penyeimbangan kelas menggunakan **SMOTE** (Synthetic Minority Over-sampling Technique) dan **ADASYN** (Adaptive Synthetic Sampling).

### 1. Algoritma yang Digunakan

- **Decision Tree**: 
  - Decision Tree adalah algoritma yang membangun model prediktif berbentuk pohon dengan memecah dataset menjadi subset yang lebih kecil berdasarkan fitur yang paling relevan. Keunggulan dari Decision Tree adalah kemampuannya untuk memberikan interpretasi yang jelas tentang keputusan yang diambil, serta kecepatan dalam pelatihan dan prediksi. Namun, Decision Tree juga rentan terhadap overfitting, terutama pada dataset yang kecil atau tidak seimbang.

- **Random Forest**: 
  - Random Forest adalah ensemble learning method yang menggabungkan beberapa Decision Trees untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dalam Random Forest dilatih pada subset acak dari data, dan keputusan akhir ditentukan dengan mayoritas suara. Ini membuat Random Forest lebih stabil dan mampu menangani dataset dengan lebih baik.

### 2. Metode Penyeimbangan Kelas

- **SMOTE**: 
  - SMOTE meningkatkan kelas minoritas dengan membuat contoh sintetis di antara titik-titik data yang sudah ada. Ini membantu meningkatkan jumlah contoh dalam kelas minoritas, sehingga model tidak bias terhadap kelas mayoritas.

- **ADASYN**: 
  - ADASYN adalah pengembangan dari SMOTE yang tidak hanya menciptakan data sintetis tetapi juga mempertimbangkan kesulitan dari kelas minoritas. Ini berfokus pada contoh yang lebih sulit untuk diklasifikasikan, sehingga meningkatkan representasi dari titik data yang lebih kompleks dalam kelas minoritas.

### 3. Evaluasi Model

Kami mengevaluasi model berdasarkan beberapa metrik berikut:

- **Akurasi**: Persentase dari prediksi yang benar dibandingkan dengan total prediksi.
- **Precision**: Proporsi dari prediksi positif yang benar dari semua prediksi positif.
- **Recall**: Proporsi dari prediksi positif yang benar dari semua data positif yang sebenarnya.
- **F1 Score**: Rata-rata harmonis dari precision dan recall, memberikan gambaran yang lebih baik tentang kinerja model pada dataset yang tidak seimbang.

### 4. Hasil Model

Setelah menjalankan model dengan dataset asli dan teknik penyeimbangan (SMOTE dan ADASYN), kami memperoleh hasil sebagai berikut:

- **Tanpa Penyeimbangan**:
  - Decision Tree menunjukkan akurasi 0.9017 dengan precision 0.9069, recall 0.9017, dan F1 score 0.9037.
  - Random Forest memiliki performa lebih baik dengan akurasi 0.9302, precision 0.9288, recall 0.9302, dan F1 score 0.9293.

- **Dengan SMOTE**:
  - Decision Tree memiliki akurasi 0.9218, precision 0.9225, recall 0.9218, dan F1 score 0.9217.
  - Random Forest menunjukkan kinerja yang lebih baik dengan akurasi 0.9483, precision 0.9484, recall 0.9483, dan F1 score 0.9480.

- **Dengan ADASYN**:
  - Decision Tree memiliki akurasi 0.8758, precision 0.8753, recall 0.8758, dan F1 score 0.8750.
  - Random Forest menunjukkan hasil yang lebih baik dengan akurasi 0.9385, precision 0.9384, recall 0.9385, dan F1 score 0.9382.

### 5. Diskusi Teknis

Dari hasil yang diperoleh, dapat dilihat bahwa Random Forest selalu mengungguli Decision Tree dalam hal akurasi dan metrik evaluasi lainnya. Ini menunjukkan bahwa pendekatan ensemble lebih efektif dalam mengatasi masalah klasifikasi, terutama pada dataset yang mungkin tidak seimbang.

Penggunaan SMOTE dan ADASYN memberikan hasil yang beragam; dalam beberapa kasus, peningkatan signifikan terlihat pada akurasi dan metrik lainnya. Namun, penerapan teknik penyeimbangan juga menunjukkan bahwa model dapat lebih baik atau bahkan lebih buruk tergantung pada algoritma dan bagaimana mereka beradaptasi dengan data yang diberikan.

Akhirnya, pemilihan algoritma dan metode penyeimbangan kelas sangat penting untuk meningkatkan kinerja model, terutama dalam situasi di mana data tidak seimbang.


## Evaluation

Setelah melatih model menggunakan dataset pelatihan dan melakukan prediksi pada dataset pengujian, kami melanjutkan dengan langkah-langkah evaluasi untuk menganalisis kinerja setiap model. Proses evaluasi ini mencakup beberapa langkah kunci:

### Metrik Evaluasi:

- **Classification Report**: Menggunakan `classification_report` dari scikit-learn, kami dapat memperoleh metrik seperti precision, recall, dan F1 score untuk setiap kelas. Ini memberikan gambaran menyeluruh tentang kinerja model di setiap kategori target.
  
- **Confusion Matrix**: Kami menggunakan matriks kebingungan untuk visualisasi prediksi model dibandingkan dengan label sebenarnya. Ini membantu dalam memahami di mana model melakukan kesalahan, apakah lebih banyak prediksi positif yang salah (False Positives) atau prediksi negatif yang salah (False Negatives).
  
- **ROC AUC Score**: Kami menghitung ROC AUC untuk menganalisis kemampuan model dalam membedakan antara kelas positif dan negatif. Ini dilakukan dengan menghitung False Positive Rate dan True Positive Rate untuk setiap kelas, dan kemudian menghitung area di bawah kurva ROC (AUC).

### Visualisasi:

- **Plot Confusion Matrix**: Kami menggunakan heatmap dari seaborn untuk memvisualisasikan matriks kebingungan, sehingga memberikan pemahaman yang lebih baik tentang prediksi model.
  
- **Plot ROC Curve**: ROC curve membantu dalam menganalisis trade-off antara sensitivity dan specificity untuk setiap kelas.

### Hasil Evaluasi:

Setiap metrik yang dihasilkan akan dicatat dan dibandingkan antar model untuk menentukan model mana yang memberikan kinerja terbaik.


## Conclusion

Studi ini memiliki implikasi penting dalam upaya menurunkan angka kematian ibu dan anak, terutama di negara-negara dengan sumber daya terbatas. Dengan pemanfaatan teknologi yang terjangkau seperti Cardiotocogram (CTG), diharapkan para tenaga medis dapat lebih cepat dan akurat dalam mendeteksi komplikasi kehamilan, sehingga intervensi medis dapat dilakukan lebih awal. 

Hasil penelitian ini juga berpotensi untuk memberikan kontribusi dalam pengembangan protokol klinis yang lebih baik dalam pemantauan kehamilan. Cardiotocogram menyediakan informasi yang penting dan mudah diakses mengenai kesehatan janin, yang dapat membantu mengurangi angka kematian ibu dan anak. 

Dengan pemanfaatan teknologi ini, serta dukungan dari analisis berbasis data, diharapkan proses deteksi dini masalah kesehatan janin dapat lebih ditingkatkan. Melalui pendekatan berbasis data dan teknologi, kita dapat meningkatkan hasil kesehatan bagi ibu dan anak, serta mewujudkan sistem kesehatan yang lebih efisien dan responsif.




