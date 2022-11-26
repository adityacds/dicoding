# Laporan Proyek *Machine Learning* – Aditya Chandra D.S

## Project Overview
Buku adalah jendela dunia. Dengan buku seseorang dapat menjelajah ke dunia luar tanpa perlu pergi ke dunia luar. Dengan buku seseorang dapat memperoleh pengetahuan yang tiada batas, melintas waktu, dan mengenal seseorang dari seluruh belahan dunia, karena buku merupakan sumber ilmu pengetahuan [1].
Kegiatan membaca buku sangat penting bagi kehidupan manusia, dengan terbiasa membaca buku maka seseorang akan memiliki cakrawala pengetahuan yang luas [2].
Namun dengan banyaknya jumlah buku yang tersedia terkadang membuat pembaca kebingungan dalam menentukan buku yang hendak mereka baca. Terkadang dijumpai pembaca yang hanya ingin membaca buku-buku yang dengan reputasi penjualan terbaik. Ada pula pembaca yang menentukan buku-buku yang akan dibaca selanjutnya berdasarkan rating dari buku-buku yang telah dilihatnya. [3]

Berdasarkan permasalahan tersebut, pada proyek ini akan dibuat suatu model sistem rekomendasi menggunakan teknik *collaborative filtering* untuk merekomendasikan buku-buku yang mungkin akan dibaca oleh pengguna.

## Business Understanding

**Problem Statement** \
Berdasarkan *Project Overview* yang sudah dipaparkan sebelumnya, berikut masalah yang dapat diselesaikan dalam proyek ini :
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

* Menggabungkan dataset books, ratings, dan users. Untuk memudahkan pemrosesan, maka ketiga dataset tersebut digabungkan terlebih dahulu.

Tabel 4. Penggabungan dataset
    
|   |       ISBN |          Book-Title |          Book-Author | Year-Of-Publication |               Publisher |                                       Image-URL-S |                                       Image-URL-M |                                       Image-URL-L | User-ID | Book-Rating |                  Location |  Age |
|--:|-----------:|--------------------:|---------------------:|--------------------:|------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------:|------------:|--------------------------:|-----:|
| 0 | 0195153448 | Classical Mythology |   Mark P. O. Morford |                2002 | Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... |     2.0 |         0.0 | stockton, california, usa | 18.0 |
| 1 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... |     8.0 |         5.0 |  timmins, ontario, canada |  NaN |
| 2 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 11400.0 |         0.0 |   ottawa, ontario, canada | 49.0 |
| 3 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 11676.0 |         8.0 |             n/a, n/a, n/a |  NaN |
| 4 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 41385.0 |         0.0 |  sudbury, ontario, canada |  NaN |    
    
* Melihat jumlah baris dan kolom pada dataset yang telah digabungkan.
* Mengecek jumlah data kosong pada setiap kolom.

Tabel 5. Jumlah data yang kosong

| ISBN                | 0      |
|---------------------|--------|
| Book-Title          | 0      |
| Book-Author         | 1      |
| Year-Of-Publication | 0      |
| Publisher           | 2      |
| Image-URL-S         | 0      |
| Image-URL-M         | 0      |
| Image-URL-L         | 4      |
| User-ID             | 1209   |
| Book-Rating         | 1209   |
| Location            | 1209   |
| Age                 | 279044 |
| dtype: int64        |        |

Berdasarkan tabel 5, diketahui bahwa ada yang datanya kosong.

* Mengecek apakah ada data yang terduplikat. Ternyata tidak ada data yang terduplikat.
* Melihat visualisasi distribusi rating buku.

![image](https://user-images.githubusercontent.com/65145111/204081306-a1861ce1-6d21-4944-bbbb-0789821893e7.png)
Gambar 1. Visualisasi distribusi rating buku.

Berdasarkan gambar 1, ditemukan bahwa data rating tidak seimbang. Oleh karena itu pada tahap selanjutnya dilakukan penyeimbangan data

* Menghapus kategori yang tidak diperlukan pada kolom. kategori yang dihapus adalah rating dengan nilai 0 pada kolom Book-Rating.
* Mengecek visualisasi distribusi rating buku yang sudah ditangani data tidak seimbangnya.

![image](https://user-images.githubusercontent.com/65145111/204081385-4f09cf33-613d-4f43-88bd-7a8978a90f27.png)
Gambar 2. Visualisasi distribusi rating buku setelah penyeimbangan data.

Berdasarkan gambar 2, ditemukan bahwa data rating sudah seimbang.

* Menghapus data yang mempunyai nilai *null*.

Tabel 6. Jumlah data yang memiliki nilai *null*

| ISBN                | 0 |
|---------------------|---|
| Book-Title          | 0 |
| Book-Author         | 0 |
| Year-Of-Publication | 0 |
| Publisher           | 0 |
| Image-URL-S         | 0 |
| Image-URL-M         | 0 |
| Image-URL-L         | 0 |
| User-ID             | 0 |
| Book-Rating         | 0 |
| Location            | 0 |
| Age                 | 0 |
| dtype: int64        |   |

Berdasarkan data pada tabel 6, diketahui bahwa tidak ada data yang bernilai null.

* Melakukan encoding, di antaranya adalah: menyandikan (encode) fitur 'UserID' dan fitur 'ISBN' ke dalam indeks integer, memetakan ‘User-ID’ dan ‘ISBN’ ke dataframe yang berkaitan, kemudian mengecek beberapa hal dalam data seperti jumlah user, jumlah isbn, kemudian mengubah nilai rating menjadi float.
* Melakukan pengacakan data. Hal ini dilakukan agar distribusi data menjadi acak.
* Membagi dataset menjadi dua bagian, yaitu 80% untuk data training, dan 20% untuk data validasi. Tahap ini menggunakan library sklearn.
    
# Modeling

Setelah melakukan data preparation, langkah selanjutnya yang dilakukan adalah membuat model *machine learning*. Dalam penyusunan sistem rekomendasi ini, penulis menggunakan metode collaborative filtering yang dibuat berdasarkan rating buku yang ditelah diberikan oleh user.

Pada tahap modelling ini, model menghitung skor kecocokan antara user dan ISBN dengan teknik embedding. Berikut tahapan yang dilakukan :
- Melakukan proses embedding terhadap data user dan ISBN.
- Melakukan operasi perkalian dot product antara embedding user dan ISBN.
- Menambahkan bias untuk setiap user dan ISBN. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
- Membuat class RecommenderNet dengan keras Model class.
- Proses compile terhadap model menggunakan Binary Crossentropy untuk menghitung loss function, Adam sebagai optimizer, dan RMSE sebagai metrics evaluation.

Top 10 book recommendation dapat dilihat pada tabel 7.

Tabel 7. Top 10 book recommendation

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

Setelah membangun model *machine learning*, kemudian dilakukan evaluasi kinerja model yang dihasilkan dengan menggunakan menggunakan metrik RMSE (Root Mean Square Error). Root Mean Square Error (RMSE) adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. Cara Menghitung Root Mean Square Error (RMSE) adalah dengan mengurangi nilai aktual dengan nilai prediksi kemudian dikuadratkan dan dijumlahkan keseluruhan hasilnya kemudian dibagi dengan banyaknya data. Hasil perhitungan tersebut selanjutnya dihitung kembali untuk mencari nilai dari akar kuadrat [4].

Visualisasi metrik RMSE dapat dilihat pada gambar 3. 

![image](https://raw.githubusercontent.com/adityacds/dicoding/main/metrics.png) \
Gambar 3. Visualisasi metrik RMSE 

# Kesimpulan

Model *Machine Learning* berupa sistem rekomendasi buku bagi pengguna menggunakan Collaborative Filtering telah selesai dibuat. Setelah diujikan, model ini bekerja cukup baik dalam memberikan 10 rekomendasi teratas terhadap buku berdasarkan preferensi pengguna sebelumnya. 10 buku rekomendasi teratas bisa dilihat pada tabel 7.


# Referensi

[1] Gresi A.R., Alan N., Khasanah B.R., Robby A.S., Priyadi N.P. (2013). Rumah Baca Jendela Dunia, Sebuah Model Perpustakaan Panti Asuhan. Jurnal Ilmiah Mahasiswa, Vol. 3 No.2. https://media.neliti.com/media/publications/96720-ID-rumah-baca-jendela-dunia-sebuah-model-pe.pdf

[2] Shofaussamawati. (2014). Menumbuhkan Minat Baca dengan Pengenalan Perpustakaan Pada Anak Sejak Dini. Jurnal IAIN, Vol. 2 No.1. https://journal.iainkudus.ac.id/index.php/Libraria/article/download/1189/1082

[3] Ritdrix, A.H. (2018). Sistem Rekomendasi Buku Menggunakan Metode Item-Based Collaborative Filtering. Universitas Diponegoro. http://eprints.undip.ac.id/65823/1/laporan_24010311130044_1.pdf

[4] Khoiri. (2020). _Pengertian dan Cara Menghitung Root Mean Square Error (RMSE). https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html
