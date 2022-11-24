# Laporan Proyek *Machine Learning* – Aditya Chandra D.S

## Domain Project
Domain proyek yang dipilih dalam proyek *machine learning* yang pertama ini adalah mengenai kesehatan dengan judul proyek “Prediksi Diagnosis Penyakit Stroke Pada Manusia”

## Latar Belakang
Stroke didefinisikan sebagai gangguan suplai darah pada otak yang biasanya disebabkan karena pecahnya pembuluh darah atau sumbatan oleh gumpalan darah. Hal ini menyebabkan gangguan pasokan oksigen dan nutrisi di otak sehingga terjadi kerusakan pada jaringan otak (WHO,2016).
Data Riskesdas tahun 2018 menyebutkan prevalensi stroke di Indonesia pada usia ≥ 15 tahun adalah 10,9% per 1000 penduduk, sementara pada tahun 2013 angka prevalensi stroke sebanyak 7% sehingga ada peningkatan sebesar 3,9% selama kurun waktu 5 tahun. Daerah Istimewa Yogyakarta (DIY) memiliki prevalensi stroke tertinggi di tahun 2018 sebesar 14,7% (Kementrian Kesehatan Repoblik Indonesia, 2018). 
Diperlukan kesadaran bagi setiap orang tentang bahaya penyakit stroke ini, karena masalah ini tidak hanya menyerang diusia tertentu saja. Oleh karena itu maka dibuatlah sebuah model *machine learning* untuk memprediksi apakah seseorang terkena penyakit stroke atau tidak. Dengan adanya model *machine learning* ini diharapkan kita dapat mengindetifikasi penyakit stroke lebih awal.

## Business Understanding

**Problem Statement** \
Berdasarkan latar belakang yang sudah dipaparkan sebelumnya, berikut masalah yang dapat diselesaikan dalam proyek ini : \
Bagaimana membuat model untuk memprediksi penyakit stroke pada manusia menggunakan Teknik *machine learning*?

**Goals** \
Membuat model *machine learning* untuk memprediksi penyakit stroke pada manusia.

**Solution Statements** \
Solusi yang dilakukan untuk mencapai tujuan dari proyek ini adalah : 
  -	Membandingkan algoritma *AdaBoost*, *Gradient Boosting*, dan *Random Forest*
  -	Menggunakan metrik pengukuran akurasi, *precision*, *recall*, dan *f1-score*.

## Data Understanding

Dataset yang digunakan pada proyek ini diambil dari\
https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset

Penjelasan Variabel pada *Brain Stroke dataset* adalah sebagai berikut:
  -	*gender*: Menyatakan Jenis Kelamin (*"Male", "Female", "Other"*)
  -	*age*: Menyatakan umur dari pasien
  -	*hypertension*: Menyatakan apakah pasien memiliki hipertensi atau tidak (0 jika pasien tidak memiliki hipertensi, 1 jika pasien memiliki hipertensi)
  -	*heart disease*: Menyatakan apakah pasien memiliki penyakit jantung atau tidak (0 jika pasien tidak memiliki penyakit jantung, 1 jika pasien memiliki penyakit      jantung)
  -	*ever married*: Menyatakan apakah pasien sudah pernah menikah atau tidak (0 jika pasien tidak pernah menikah, 1 jika pasien pernah menikah)
  -	*work type*: Menyatakan Jenis Pekerjaan (*"children", "govt job", "never worked", "private", "self-employeed"*)
  -	*residence type*: Menyatakan tempat tinggal (*"rular", "urban"*)
  -	*avg glucose level*: Menyatakan kadar gula rata-rata dalam darah
  -	*bmi*: Menyatakan indeks masa tubuh (*body mass index*)
  -	*smoking status*: Menyatakan apakah pernah merokok atau tidak (*"formerly smoked", "never smoked", "smokes", "Unknown"*)
  -	*stroke*: Menyatakan apakah terdiagnosis stroke (0 jika pasien tidak memiliki stroke, 1 jika pasien memiliki stroke)

## Exploratory Data Analysis

Penulis mengelompokan umur pasien menjadi 0-25, 25-40, 40-60, dan >60. Berikut merupakan distribusi data pasien berdasarkan variabel dataset:

![image](https://user-images.githubusercontent.com/65145111/203143337-e417ce1f-ea61-418e-8cff-bb3efa599a09.png) \
Gambar 1. Distribusi data pasien.

Dari gambar tersebut, dapat dilihat bahwa terdapat ketidakseimbangan dataset, khususnya pada variabel target stroke. Hal ini dapat memengaruhi model 
*learning* yang akan dibahas lebih lanjut saat evaluasi.

## Data Preparation

*	Mengubah data kategorikal menjadi data yang dimengerti mesin, yaitu angka.
    1. Mengubah gender dan ever married menjadi 0 dan 1.
    2. Melakukan one hot encoding untuk data kategorikal lainnya, yaitu *age, work type, residence type,* dan *smoking status.*
*	Melakukan data splitting menjadi data training dan data testing dengan perbandingan 80:20\
  Tahapan ini bertujuan agar model yang dilatih dapat diuji dengan data yang berbeda dari data yang digunakan dalam pelatihan. Pada proyek ini data dibagi menjadi dua dengan persentase untuk training sebesar 80% dan sisanya 20% untuk testing. Fungsi train_test_split pada library sklearn yang akan digunakan untuk menangani tahapan ini.
*	Melakukan standarisasi data
  Melakukan standardisasi data pada semua fitur data. Hal ini dilakukan untuk membuat semua fitur berada dalam skala data yang sama yaitu dengan range 0-1.      Strandadisasi data ini menggunakan fungsi *StandardScaler*. Rumus fungsi StandardScaler : \
  $$ z = {(x-u) \over s} $$ \
  Di mana z adalah nilai baru, x adalah nilai asli, u adalah mean dan s adalah standar deviasi.
  
## Modeling
Setelah melakukan tahapan preprocessing makan data telah siap dimasukan kedalam model. Pada proyek ini dilakukan pembuatan model dengan dua cara, yaitu tanpa menyeimbangkan data dan dengan menyeimbangkan data (menggunakan random undersampling). Penulis menggunakan 3 model *machine learning* yang berbeda, yaitu *AdaBoost*, *Gradient Boosting*, dan *Random Forest*. Parameter yang digunakan pada kedua cara tersebut sama, yaitu :
  1.	*AdaBoost* \
        Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah *(weak learners)* sehingga membentuk suatu model yang kuat (strong ensemble learner). Dalam mengimplementasikan algoritma ini, saya menggunakan method AdaBoostClassifier dari sklearn.ensemble dengan parameter n_estimators=50, learning_rate=1 dan random_state=42.
        
        
  2.	*Gradient Boosting* \
      Sama seperti *AdaBoost*,  Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Dalam mengimplementasikan algoritma ini, saya menggunakan method GradientBoostingClassifier dari sklearn.ensemble dengan parameter learning_rate=1 dan random_state=42. 
        
  3.	*Random Forest* \
      Dalam mengimplementasikan algoritma ini, saya menggunakan method RandomForestClassifier dari sklearn.ensemble dengan parameter n_estimators=50 dan learning_rate=1, dan random_state=42. 
      Kelebihan dari algoritma yang ini adalah dapat memperkiraan variabel apa yang penting dalam klasifikasi, sedangkan kekurangan dari algoritma ini yaitu memiliki kompleksitas yang tinggi.

## Evaluation

Metrik yang digunakan ada 4, yaitu:
  1.	*Accuracy*
  2.	*Precision*
  3.	*Recall*
  4.	*F1 Score*

Berikut merupakan rumus dari keempat metrik tersebut:

$$ Accuracy = {(TP + TN) \over (TP + TN + FP + FN)} $$ \
$$ Precision = {TP \over (TP + FP)} $$ \
$$ Recall = {TP \over (TP + FN)}$$ \
$$ F1 Score = {2 * (Precision * Recall) \ (Precision + Recall)} \

Pada permasalahan ini, mendeteksi pasien dengan stroke sangatlah penting. Oleh karena itu, metrik paling sesuai untuk masalah ini adalah *recall* dan berusaha untuk meminimalkan jumlah *false negative*. Adapun jumlah *false positive* adalah prioritas kedua untuk diminimalisir.
Hasil dari pelatihan model pertama kali dapat dilihat pada Tabel 1.

Tabel 1. Hasil pelatihan model pertama kali.
|                   | Accuracy | Precision |   Recall | F1-Score |
|------------------:|---------:|----------:|---------:|---------:|
|      AdaBoost     | 0.945838 |       0.0 |      0.0 |      0.0 |
| Gradient Boosting | 0.929789 |  0.214286 | 0.111111 | 0.146341 |
|   Random Forest   | 0.942828 |       0.0 |      0.0 |      0.0 |


![image](https://user-images.githubusercontent.com/65145111/203688321-c0508a9c-bb6c-4cae-a177-2a1fa76800dc.png) \
Gambar 2. Ada Boost

![image](https://user-images.githubusercontent.com/65145111/203688337-d77cfeff-44ed-4454-a111-d6a50be628fb.png) \
Gambar 3. Gradient Boosting

![image](https://user-images.githubusercontent.com/65145111/203688356-801ed81c-584e-444b-b728-80269d8259d6.png)) \
Gambar 4. Random Forest.

Dapat dilihat bahwa kelima model memiliki nilai *recall* yang rendah dengan banyak nilai *false negative*. Hal ini kemungkinan diakibatkan karena data yang tidak imbang di mana terdapat 248 pasien mengalami stroke dan 4733 data pasien tanpa stroke. Oleh karena itu, untuk menyeimbangkan data dilakukanlah *random undersampling*.
Hasil pelatihan model dengan data yang seimbang dapat dilihat pada tabel 2.

Tabel 2. Hasil pelatihan model pertama dengan data yang seimbang.

|                   | Accuracy | Precision | Recall | F1-Score |
|------------------:|---------:|----------:|-------:|---------:|
|      AdaBoost     |     0.67 |  0.660377 |    0.7 | 0.679612 |
| Gradient Boosting |     0.62 |  0.607143 |   0.68 | 0.641509 |
|   Random Forest   |     0.65 |  0.631579 |   0.72 | 0.672897 |


![image](https://user-images.githubusercontent.com/65145111/203688233-eee74955-835a-4d55-b9f9-3cabe10cc7bb.png) \
Gambar 5. Ada Boost

![image](https://user-images.githubusercontent.com/65145111/203688200-c6b4235b-3f58-48b7-9546-203912975dd1.png) \
Gambar 6. Gradient Boosting

![image](https://user-images.githubusercontent.com/65145111/203688170-988eef64-2772-4a83-a763-839fd132a855.png) \
Gambar 7. Random Forest

Dapat dilihat bahwa ketiga model memiliki nilai *recall* yang jauh lebih baik dari model yang sebelumnya. Di mana model terbaik adalah model *Random Forest* dengan nilai *recall* 0.72 dan jumlah *false negative* terkecil yaitu 14.

Berdasarkan percobaan yang telah dilakukan, dalam masalah ini penulis memilih *Random Forest* sebagai model terbaik.
