# Laporan Proyek *Machine Learning* – Aditya Chandra D.S

## Domain Project
Domain proyek yang dipilih dalam proyek *machine learning* yang pertama ini adalah mengenai kesehatan dengan judul proyek “Prediksi Diagnosis Penyakit Stroke Pada Manusia”

## Latar Belakang
Stroke didefinisikan sebagai gangguan suplai darah pada otak yang biasanya disebabkan karena pecahnya pembuluh darah atau sumbatan oleh gumpalan darah. Hal ini menyebabkan gangguan pasokan oksigen dan nutrisi di otak sehingga terjadi kerusakan pada jaringan otak (WHO,2016).
Data Riskesdas tahun 2018 menyebutkan prevalensi stroke di Indonesia pada usia ≥ 15 tahun adalah 10,9% per 1000 penduduk, sementara pada tahun 2013 angka prevalensi stroke sebanyak 7% sehingga ada peningkatan sebesar 3,9% selama kurun waktu 5 tahun. Daerah Istimewa Yogyakarta (DIY) memiliki prevalensi stroke tertinggi di tahun 2018 sebesar 14,7% (Kementrian Kesehatan Repoblik Indonesia, 2018). 
Diperlukan kesadaran bagi setiap orang tentang bahaya penyakit stroke ini, karena masalah ini tidak hanya menyerang diusia tertentu saja. Oleh karena itu maka dibuatlah sebuah model *machine learning* untuk memprediksi apakah seseorang terkena penyakit stroke atau tidak. Dengan adanya model *machine learning* ini diharapkan kita dapat mengindetifikasi penyakit stroke lebih awal.

## Business Understanding

**Problem Statement**
Berdasarkan latar belakang yang sudah dipaparkan sebelumnya, berikut masalah yang dapat diselesaikan dalam proyek ini :
Bagaimana membuat model untuk memprediksi penyakit stroke pada manusia menggunakan Teknik *machine learning*?

**Goals**
Membuat model *machine learning* untuk memprediksi penyakit stroke pada manusia.

**Solution Statements**
Solusi yang dilakukan untuk mencapai tujuan dari proyek ini adalah :
  -	Membandingkan algoritma *AdaBoost*, *Gradient Boosting*, *Random Forest*, *Decision Tree*, dan *SVM*
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
  Melakukan standardisasi data pada semua fitur data. Hal ini dilakukan untuk membuat semua fitur berada dalam skala data yang sama yaitu dengan range 0-1.      Strandadisasi data ini menggunakan fungsi *StandardScaler*. Berikut rumus dari *Standardscaler* :\
  $$ z = {x - b \over 2a} $$
  z = (x – u ) / s
  Di mana z adalah nilai baru, x adalah nilai asli, u adalah mean dan s adalah standar deviasi.
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
## Modeling

Penulis menggunakan 5 model *machine learning* yang berbeda, yaitu:
  1.	*AdaBoost*
  2.	*Gradient Boosting*
  3.	*Random Forest*
  4.	*Decision Tree*
  5.	*SVM*
  
Semua model dilatih menggunakan parameter *default* yang disediakan *library sklearn*.

## Evaluation

Metrik yang digunakan ada 4, yaitu:
  1.	*Accuracy*
  2.	*Precision*
  3.	*Recall*
  4.	*F1 Score*

Pada permasalahan ini, mendeteksi pasien dengan stroke sangatlah penting. Oleh karena itu, metrik paling sesuai untuk masalah ini adalah *recall* dan berusaha untuk meminimalkan jumlah *false negative*. Adapun jumlah *false positive* adalah prioritas kedua untuk diminimalisir.
Hasil dari pelatihan model pertama kali dapat dilihat pada Tabel 1.

Tabel 1. Hasil pelatihan model pertama kali.
|                     | Accuracy | Precision |  Recall | F1 Score |
|---------------------|:--------:|:---------:|:-------:|:--------:|
| AdaBoost            | 0.945838 | 0         | 0       | 0        |
| Gradient   Boosting | 0.945838 | 0.5       | 0.01852 | 0.035714 |
| Random   Forest     | 0.942828 | 0         | 0       | 0        |
| Decision   Tree     | 0.90672  | 0.157895  | 0.16667 | 0.162162 |
| SVM                 | 0.945838 | 0         | 0       | 0        |

![image](https://user-images.githubusercontent.com/65145111/203279922-5c6c7ded-886c-446b-bffa-e66bf308da09.png) \
Gambar 2. Ada Boost

![image](https://user-images.githubusercontent.com/65145111/203280003-059d9a5b-7d9e-4c4d-b168-e243e11ff2d5.png) \
Gambar 3. Gradient Boosting

![image](https://user-images.githubusercontent.com/65145111/203280884-6800243c-85be-4529-b59f-76ece4d37872.png) \
Gambar 4. Random Forest.

![image](https://user-images.githubusercontent.com/65145111/203280971-a22a91fd-6fcf-4993-9e4d-5dd675fc4473.png) \
Gambar 5. Decision Tree

![image](https://user-images.githubusercontent.com/65145111/203281054-79c91828-932d-46ee-a023-219de7da4e6b.png) \
Gambar 6. SVM

Dapat dilihat bahwa kelima model memiliki nilai *recall* yang rendah dengan banyak nilai *false negative*. Hal ini kemungkinan diakibatkan karena data yang tidak imbang di mana terdapat 248 pasien mengalami stroke dan 4733 data pasien tanpa stroke. Oleh karena itu, untuk menyeimbangkan data dilakukanlah *random undersampling*.
Hasil pelatihan model dengan data yang seimbang dapat dilihat pada tabel 2.

Tabel 2. Hasil pelatihan model pertama dengan data yang seimbang.

|                     | Accuracy | Precision | Recall | F1 Score |
|---------------------|:--------:|:---------:|:------:|:--------:|
| AdaBoost            | 0.67     | 0.660377  | 0.7    | 0.67961  |
| Gradient   Boosting | 0.61     | 0.603774  | 0.64   | 0.621359 |
| Random   Forest     | 0.66     | 0.637931  | 0.74   | 0.68519  |
| Decision   Tree     | 0.58     | 0.571429  | 0.64   | 0.603774 |
| SVM                 | 0.73     | 0.72549   | 0.74   | 0.73267  |

![image](https://user-images.githubusercontent.com/65145111/203283286-3121cbb1-1c0f-4f70-b61e-d5d1895deed4.png) \
Gambar 7. Ada Boost

![image](https://user-images.githubusercontent.com/65145111/203283325-e828f5a7-b1b8-47dc-b1a9-ae889a0d4177.png) \
Gambar 8. Gradient Boosting

![image](https://user-images.githubusercontent.com/65145111/203283361-0f4543c6-3e61-47c9-93f0-14f79bc2a2a3.png) \
Gambar 9. Random Forest

![image](https://user-images.githubusercontent.com/65145111/203283407-285c75d8-013b-4a75-bcc7-50134df13d32.png) \
Gambar 10. Decision Tree

![image](https://user-images.githubusercontent.com/65145111/203283445-dd66079c-abf9-4030-9432-25d1cd5febb7.png) \
Gambar 11. SVM

Dapat dilihat bahwa kelima model memiliki nilai *recall* yang jauh lebih baik dari model yang sebelumnya. Di mana model terbaik adalah model *Random Forest* dan *SVM* dengan nilai *recall* 0.74 dan jumlah *false negative* terkecil yaitu 13.

Berdasarkan percobaan yang telah dilakukan, dalam masalah ini penulis memilih 2 model sebagai model terbaik, yaitu *Random Forest* dan *SVM*.
