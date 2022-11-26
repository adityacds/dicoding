# Laporan Proyek *Machine Learning* – Aditya Chandra D.S

## Project Overview
Buku adalah jendela dunia. Dengan buku seseorang dapat menjelajah ke dunia luar tanpa perlu pergi ke dunia luar. Dengan buku seseorang dapat memperoleh pengetahuan yang tiada batas, melintas waktu, dan mengenal seseorang dari seluruh belahan dunia, karena buku merupakan sumber ilmu pengetahuan [1].
Kegiatan membaca buku sangat penting bagi kehidupan manusia, dengan terbiasa membaca buku maka seseorang akan memiliki cakrawala pengetahuan yang luas [2].
Namun dengan banyaknya jumlah buku yang tersedia terkadang membuat pembaca kebingungan dalam menentukan buku yang hendak mereka baca. Terkadang dijumpai pembaca yang hanya ingin membaca buku-buku yang dengan reputasi penjualan terbaik. Ada pula pembaca yang menentukan buku-buku yang akan dibaca selanjutnya berdasarkan rating dari buku-buku yang telah dilihatnya. [3]

Berdasarkan permasalahan tersebut, pada proyek ini akan dibuat suatu model sistem rekomendasi menggunakan teknik *collaborative filtering* untuk merekomendasikan buku-buku yang mungkin akan dibaca oleh pengguna.

## Business Understanding

**Problem Statement** \
Berdasarkan *Project Overview* yang sudah dipaparkan sebelumnya, berikut masalah yang dapat diselesaikan dalam proyek ini : \
Bagaimana cara membuat sistem rekomendasi buku yang mungkin disukai dan belum pernah dibaca oleh pengguna?

**Goals** \
Adapun tujuan proyek ini adalah:
Memberikan rekomendasi buku yang mungkin disukai dan belum pernah dibaca oleh pengguna.

## Solution Approach
Pada proyek ini, penulis akan membuat sistem rekomendasi dengan menggunakan *collaborative filtering. Collaborative Filtering* adalah sebuah metode yang merekomendasikan item berdasarkan kemiripan user dalam hal memilih atau memberi nilai kepada item.

# Data Understanding

Dataset yang digunakan pada proyek ini adalah *Book Recommendation Dataset* yang diambil dari https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
Dataset ini memiliki 3 file csv, yaitu Books.csv, Ratings.csv, dan Users.csv.

1. Books.csv

Penjelasan mengenai variabel yang ada pada dataset Books.csv dapat dilihat pada Tabel 1.

Tabel 1. Variabel pada Books.csv

| Kolom               | Keterangan                                                                |
|---------------------|---------------------------------------------------------------------------|
| ISBN                | International Standard Book Number merupakan kode unik masing masing buku |
| Book-title          | Judul dari tiap buku                                                      |
| Book-Author         | Penulis atau pengarang tiap buku                                          |
| Year-Of_Publication | Tahun terbit tiap buku                                                    |
| Publisher           | Lembaga penerbit tiap buku                                                |
| Image-URL-S         | Link foto dari tiap buku yang berukuran kecil                             |
| Image-URL-M         | Link foto dari tiap buku yang berukuran sedang                            |
| Image-URL-L         | Link foto dari tiap buku yang berukuran besar                             |

2. Ratings.csv

Penjelasan mengenai variabel yang ada pada dataset Ratings.csv dapat dilihat pada Tabel 2.

Tabel 2. Variabel pada Ratings.csv

| Kolom       | Keterangan                                                                |
|-------------|---------------------------------------------------------------------------|
| User-ID     | Merupakan kode unik tiap user                                             |
| ISBN        | International Standard Book Number merupakan kode unik masing masing buku |
| Book-Rating | Merupakan rating dari tiap buku                                           |

3. Users.csv

Penjelasan mengenai variabel yang ada pada dataset Users.csv dapat dilihat pada Tabel 3.

Tabel 3. Variabel pada Users.csv

| Kolom    | Keterangan                    |
|----------|-------------------------------|
| User-ID  | Merupakan kode unik tiap user |
| Location | Merupakan lokasi tiap user    |
| Age      | Merupakan umur tiap user      |

Sebelum melakukan *Data Preparation*, *dataset* perlu di load terlebih dahulu kemudian dilakukan eksplorasi terhadap *dataset* tersebut.

# Data Preparation

Berikut adalah tahapan yang dilakukan dalam proses *data preparation*:

    - Menggabungkan dataset books, ratings, dan users.
    - Melihat jumlah baris dan kolom pada dataset yang telah digabungkan.
    - Mengecek jumlah data kosong pada setiap kolom.
    - Mengecek apakah ada data yang terduplikat.
    - Melihat visualisasi distribusi rating buku (ternyata data rating tidak seimbang).
    - Menghapus kategori yang tidak diperlukan pada kolom. kategori yang dihapus adalah rating dengan nilai 0 pada kolom Book-Rating.
    - Mengecek visualisasi distribusi rating buku yang sudah ditangani data tidak seimbangnya.
    - Menghapus data yang mempunyai nilai *null*.
    - Melakukan encoding, di antaranya adalah: menyandikan (encode) fitur 'UserID' dan fitur 'ISBN' ke dalam indeks integer, memetakan ‘User-ID’ dan ‘ISBN’ ke dataframe yang berkaitan, kemudian mengecek beberapa hal dalam data seperti jumlah user, jumlah isbn, kemudian mengubah nilai rating menjadi float.
    - Melakukan pengacakan data. hal ini dilakukan agar distribusi data menjadi acak.
    - Membagi dataset menjadi dua bagian. 80% untuk data training, dan 20% untuk data validasi.
    
# Modeling

Setelah melakukan data preparation, langkah selanjutnya yang dilakukan adalah membuat model machine learning. Dalam penyusunan sistem rekomendasi ini, penulis menggunakan metode collaborative filtering yang dibuat berdasarkan rating buku yang ditelah diberikan oleh user.

Pada tahap modelling ini, model menghitung skor kecocokan antara user dan ISBN dengan teknik embedding. Berikut tahapan yang dilakukan :
- Melakukan proses embedding terhadap data user dan ISBN.
- Melakukan operasi perkalian dot product antara embedding user dan ISBN.
- Menambahkan bias untuk setiap user dan ISBN. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
- Membuat class RecommenderNet dengan keras Model class.
- Proses compile terhadap model menggunakan Binary Crossentropy untuk menghitung loss function, Adam sebagai optimizer, dan RMSE sebagai metrics evaluation.

Top 10 book recommendation dapat dilihat pada tabel 4.

Tabel 4. Top 10 book recommendation

| No | Penulis                    | Judul Buku                                                                       |
|----|----------------------------|----------------------------------------------------------------------------------|
| 1  | Pamela E. Apkarian-Russell | Postmarked Yesteryear: 30 Rare Holiday Postcards                                 |
| 2  | Philip D. Eastman          | Go, Dog, Go (I Can Read It All by Myself Beginner Books)                         |
| 3  | J. R. R. Tolkien           | The Two Towers (The Lord of the Rings, Part 2)                                   |
| 4  | Shel Silverstein           | The Giving Tree                                                                  |
| 5  | Bathroom Readers Institute | Uncle John's Supremely Satisfying Bathroom Reader (Uncle John's Bathroom Reader) |
| 6  | Art Spiegelman             | Maus 1. Mein Vater kotzt Geschichte aus. Die Geschichte eines Ã?Â?berlebenden.   |
| 7  | Scott Adams                | Dilbert: A Book of Postcards                                                     |
| 8  | Dr. Seuss                  | The Sneetches and Other Stories                                                  |
| 9  | J. K. Rowling              | Harry Potter and the Chamber of Secrets Postcard Book                            |
| 10 | Douglas Adams              | The Hitchhiker's Guide to the Galaxy                                             |

# Evaluation

Setelah membangun model machine learning, kemudian dilakukan evaluasi kinerja model yang dihasilkan dengan menggunakan menggunakan metrik RMSE (Root Mean Square Error). Root Mean Square Error (RMSE) adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. Cara Menghitung Root Mean Square Error (RMSE) adalah dengan mengurangi nilai aktual dengan nilai prediksi kemudian dikuadratkan dan dijumlahkan keseluruhan hasilnya kemudian dibagi dengan banyaknya data. Hasil perhitungan tersebut selanjutnya dihitung kembali untuk mencari nilai dari akar kuadrat [4].

Visualisasi metrik RMSE dapat dilihat pada gambar 1. 

![image](https://raw.githubusercontent.com/adityacds/dicoding/main/metrics.png) \
Gambar 1. Visualisasi metrik RMSE 

# Referensi

[1] Gresi A.R., Alan N., Khasanah B.R., Robby A.S., Priyadi N.P. (2013). Rumah Baca Jendela Dunia, Sebuah Model Perpustakaan Panti Asuhan. Jurnal Ilmiah Mahasiswa, Vol. 3 No.2. https://media.neliti.com/media/publications/96720-ID-rumah-baca-jendela-dunia-sebuah-model-pe.pdf

[2] Shofaussamawati. (2014). Menumbuhkan Minat Baca dengan Pengenalan Perpustakaan Pada Anak Sejak Dini. Jurnal IAIN, Vol. 2 No.1. https://journal.iainkudus.ac.id/index.php/Libraria/article/download/1189/1082

[3] Ritdrix, A.H. (2018). Sistem Rekomendasi Buku Menggunakan Metode Item-Based Collaborative Filtering. Universitas Diponegoro. http://eprints.undip.ac.id/65823/1/laporan_24010311130044_1.pdf

[4] Khoiri. (2020). _Pengertian dan Cara Menghitung Root Mean Square Error (RMSE). https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html
