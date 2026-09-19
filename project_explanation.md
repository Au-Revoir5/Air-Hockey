# Project Explanation - AR Air Hockey Computer Vision

## 1. Gambaran Umum Project

Project ini adalah game AR Air Hockey berbasis Computer Vision. Pemain tidak menggunakan controller fisik, tetapi mengontrol paddle melalui gesture tangan yang ditangkap oleh webcam. Kamera membaca posisi tangan secara real-time, MediaPipe mendeteksi landmark tangan, lalu model machine learning menentukan apakah gesture pemain sedang aktif atau tidak.

Secara sederhana, sistem bekerja seperti ini: jika pemain mengangkat jari telunjuk atau melakukan gesture pointing, paddle akan mengikuti posisi ujung jari. Jika gesture tidak terdeteksi sebagai pointing, paddle tidak bergerak. Dengan konsep ini, game terasa seperti air hockey virtual yang dimainkan langsung di atas tampilan kamera.

Project ini menggabungkan beberapa komponen utama:

- Computer Vision untuk membaca input dari webcam.
- MediaPipe Hands untuk mendeteksi 21 titik landmark tangan.
- Machine Learning untuk mengklasifikasikan gesture tangan.
- OpenCV untuk menampilkan kamera, menggambar elemen game, dan membuat overlay AR.
- Game physics untuk mengatur pergerakan puck, tabrakan, goal, dan skor.

## 2. Tujuan Project

Tujuan utama project ini adalah membuat game interaktif yang memanfaatkan pengenalan gesture tangan sebagai input kontrol. Selain menghasilkan game yang bisa dimainkan, project ini juga menunjukkan bagaimana pipeline Computer Vision dan Machine Learning dapat digunakan dari tahap data preparation sampai real-time inference.

Secara lebih spesifik, project ini bertujuan untuk:

- Mengenali gesture tangan berdasarkan data landmark.
- Mengubah gesture menjadi kontrol paddle di dalam game.
- Membuat sistem inference real-time yang cukup ringan untuk dijalankan melalui webcam.
- Menampilkan permainan air hockey dengan efek augmented reality, yaitu game digambar di atas background kamera.
- Mengevaluasi model machine learning agar gesture yang dipakai untuk kontrol cukup akurat.

## 3. Teknologi yang Digunakan

| Teknologi | Peran |
|---|---|
| Python | Bahasa utama untuk seluruh project |
| OpenCV | Mengakses webcam, memproses frame, dan menggambar tampilan game |
| MediaPipe Hands | Mendeteksi 21 landmark tangan dari frame kamera |
| NumPy | Perhitungan numerik seperti jarak Euclidean dan vektor physics |
| Pandas | Membaca, membersihkan, dan menyimpan dataset CSV |
| scikit-learn | Training model, preprocessing, evaluasi, dan cross-validation |
| joblib | Menyimpan dan memuat model dalam format `.pkl` |
| matplotlib | Membuat visualisasi evaluasi seperti confusion matrix |

## 4. Struktur Project

Project ini terdiri dari beberapa file utama:

| File / Folder | Fungsi |
|---|---|
| `prepare_dataset.py` | Menyiapkan dataset mentah menjadi dataset bersih untuk training |
| `train_model.py` | Melatih model gesture classifier dan memilih model terbaik |
| `test.py` | Mengecek dependency, kamera, model, dan compatibility inference |
| `air_hockey.py` | Menjalankan game AR Air Hockey secara real-time |
| `requirements.txt` | Daftar library yang dibutuhkan |
| `data/hand-gestures.csv` | Dataset mentah dari Zenodo |
| `data/gesture_data.csv` | Dataset hasil preprocessing |
| `model/gesture_model.pkl` | Model terbaik yang dipakai untuk inference |
| `model/scaler.pkl` | Scaler hasil training |
| `model/confusion_matrix.png` | Visualisasi hasil evaluasi model |
| `model/feature_importance.png` | Visualisasi feature importance atau perbandingan model |

## 5. Alur Kerja Keseluruhan

Alur kerja project dapat diringkas sebagai berikut:

```text
Dataset gesture mentah
        |
        v
prepare_dataset.py
        |
        v
Dataset bersih: gesture_data.csv
        |
        v
train_model.py
        |
        v
Model terbaik: gesture_model.pkl + scaler.pkl
        |
        v
test.py
        |
        v
air_hockey.py
        |
        v
Game AR Air Hockey berjalan secara real-time
```

Pipeline ini penting karena project tidak langsung menggunakan kamera untuk training. Training dilakukan dari dataset gesture yang sudah berbentuk fitur numerik. Setelah model dilatih, barulah model digunakan pada kamera real-time di dalam game.

## 6. Pipeline Project Secara Detail

### 6.1 Data Preparation

Tahap data preparation dilakukan oleh `prepare_dataset.py`. Tahap ini bertugas mengubah dataset mentah `data/hand-gestures.csv` menjadi dataset yang lebih rapi, bersih, dan siap dipakai untuk training.

Dataset mentah berasal dari Zenodo dengan format tanpa header. Isinya adalah label gesture dan jarak Euclidean dari landmark tangan terhadap wrist atau pergelangan tangan. Label memiliki dua kelas:

- `0`: no pointing atau closed fist, yaitu gesture tidak aktif.
- `1`: pointing atau index finger up, yaitu gesture aktif untuk menggerakkan paddle.

Dataset awal memiliki 22 kolom:

- 1 kolom label.
- 21 kolom fitur jarak landmark, yaitu `dist_0` sampai `dist_20`.

Langkah-langkah data preparation:

1. Load dataset mentah.

   File dibaca menggunakan `pd.read_csv(SRC, header=None, on_bad_lines="skip")`. Parameter `header=None` digunakan karena dataset tidak punya nama kolom. Parameter `on_bad_lines="skip"` digunakan untuk melewati baris yang formatnya rusak.

   Alasannya: dataset eksternal bisa saja memiliki baris tidak valid. Jika baris rusak langsung menyebabkan program berhenti, pipeline menjadi tidak robust. Dengan melewati bad lines, proses tetap bisa berjalan selama mayoritas data valid.

2. Memberi nama kolom.

   Kolom diberi nama `label`, lalu `dist_0` sampai `dist_20`. Penamaan ini membuat data lebih mudah dibaca dan diproses.

   Alasannya: bekerja dengan nama kolom jauh lebih jelas daripada memakai indeks angka. Misalnya, `label` lebih mudah dipahami dibanding `col[0]`.

3. Mengecek class balance.

   Script menghitung jumlah sampel untuk class 0 dan class 1. Pada dataset saat ini:

   - Class 0: 6.786 sampel.
   - Class 1: 6.784 sampel.
   - Total: 13.570 sampel.

   Dataset ini sangat seimbang karena jumlah kedua kelas hampir sama.

   Alasannya: jika dataset tidak seimbang, model bisa bias ke kelas mayoritas. Misalnya, jika 90 persen data adalah class 0, model bisa mendapat akurasi tinggi hanya dengan sering menebak class 0, padahal kemampuan mengenali class 1 buruk. Karena itu class balance perlu dicek sejak awal.

4. Menghapus `dist_0`.

   `dist_0` adalah jarak landmark 0 ke landmark 0. Landmark 0 adalah wrist, jadi `dist_0` berarti jarak wrist ke dirinya sendiri. Nilainya selalu 0.

   Alasannya: fitur yang selalu bernilai sama disebut zero-variance feature. Fitur seperti ini tidak membantu model membedakan gesture karena semua sampel memiliki nilai identik. Menghapusnya membuat input lebih ringkas, mengurangi noise, dan memastikan training fokus pada fitur yang benar-benar mengandung informasi.

5. Mengecek statistik fitur.

   Script mengecek nilai minimum, maksimum, rata-rata, dan nilai yang lebih besar dari threshold outlier. Dataset saat ini memiliki:

   - Minimum fitur: sekitar 0.0059.
   - Maksimum fitur: 1.0.
   - Rata-rata fitur: sekitar 0.5690.

   Alasannya: pengecekan statistik awal membantu mendeteksi data aneh sebelum masuk training. Karena fitur berupa jarak ternormalisasi, nilai yang terlalu besar bisa menjadi tanda data rusak atau tidak dinormalisasi dengan benar.

6. Menghapus baris dengan nilai kosong.

   Script menggunakan `dropna()` sebelum menyimpan dataset akhir. Pada dataset saat ini, jumlah nilai kosong adalah 0.

   Alasannya: banyak model scikit-learn tidak bisa menerima input `NaN`. Jika nilai kosong dibiarkan, training bisa gagal. Menghapus baris kosong adalah pilihan sederhana dan aman karena dataset masih cukup besar.

7. Menyimpan dataset bersih.

   Hasil akhir disimpan sebagai `data/gesture_data.csv`. Dataset ini berisi 20 fitur aktif, yaitu `dist_1` sampai `dist_20`, serta 1 kolom label.

   Alasannya: memisahkan dataset mentah dan dataset bersih membuat pipeline lebih jelas. Jika ingin mengulang training, model cukup membaca `gesture_data.csv` tanpa memproses ulang dataset mentah dari awal.

### 6.2 Exploratory Data Analysis (EDA)

EDA dalam project ini dilakukan secara sederhana tetapi relevan dengan kebutuhan model. Karena datanya berupa fitur numerik hasil ekstraksi landmark, EDA difokuskan pada kualitas data, distribusi kelas, dan kewajaran nilai fitur.

Tahap EDA meliputi:

1. Distribusi kelas.

   Dataset dicek apakah class 0 dan class 1 seimbang. Hasilnya hampir 50:50.

   Alasannya: distribusi kelas yang seimbang membuat akurasi lebih bisa dipercaya. Jika kelas tidak seimbang, evaluasi perlu metrik tambahan seperti recall per kelas dan mungkin teknik balancing.

2. Jumlah sampel dan jumlah fitur.

   Dataset akhir memiliki 13.570 sampel dan 20 fitur.

   Alasannya: jumlah sampel penting untuk menentukan apakah data cukup untuk training. Jumlah fitur penting untuk memastikan input model sesuai dengan fitur yang akan dibuat saat real-time inference.

3. Rentang nilai fitur.

   Fitur berada dalam rentang sekitar 0 sampai 1 karena fitur merupakan jarak yang dinormalisasi.

   Alasannya: jika rentang nilai terlalu ekstrem, model bisa belajar dari skala yang salah. Pengecekan ini juga memastikan fitur training mirip dengan fitur yang dibuat saat game berjalan.

4. Missing value.

   Dataset bersih dicek agar tidak mengandung nilai kosong.

   Alasannya: nilai kosong bisa mengganggu training dan membuat model tidak stabil.

EDA pada project ini memang tidak terlalu visual, tetapi sudah cukup untuk memastikan dataset layak dipakai. Jika ingin dikembangkan, EDA bisa ditambah histogram tiap fitur, korelasi antarfitur, dan visualisasi distribusi fitur per kelas.

### 6.3 Data Preprocessing

Preprocessing dilakukan di dua tempat: `prepare_dataset.py` dan `train_model.py`.

Di `prepare_dataset.py`, preprocessing meliputi:

- Membersihkan baris bermasalah saat load data.
- Memberi nama kolom.
- Menghapus fitur `dist_0`.
- Menghapus baris dengan `NaN`.
- Menyimpan dataset bersih.

Di `train_model.py`, preprocessing meliputi:

1. Feature-target separation.

   Dataset dipisah menjadi:

   - `X`: semua fitur `dist_1` sampai `dist_20`.
   - `y`: label gesture.

   Alasannya: model machine learning membutuhkan input fitur dan target secara terpisah. `X` dipakai untuk belajar pola, sedangkan `y` dipakai sebagai jawaban benar.

2. Train-test split.

   Data dibagi menjadi 80 persen training dan 20 persen testing menggunakan `train_test_split`.

   - Training set: 10.856 sampel.
   - Test set: 2.714 sampel.

   Parameter `stratify=y` digunakan agar proporsi class 0 dan class 1 tetap seimbang di train dan test.

   Alasannya: model harus dievaluasi pada data yang tidak digunakan saat training. Jika evaluasi dilakukan pada data training, hasilnya bisa terlalu optimis. Stratifikasi juga penting supaya test set tetap mewakili distribusi kelas asli.

3. StandardScaler.

   Fitur distandardisasi menggunakan `StandardScaler` di dalam `Pipeline`.

   Alasannya: SVM dengan RBF kernel sangat bergantung pada perhitungan jarak antar data. Jika skala fitur tidak seragam, fitur tertentu bisa mendominasi perhitungan. StandardScaler mengubah fitur agar memiliki rata-rata 0 dan standar deviasi 1.

4. Mencegah data leakage.

   `StandardScaler` dimasukkan ke dalam `Pipeline`, bukan di-fit ke seluruh dataset sebelum split atau cross-validation.

   Alasannya: data leakage terjadi ketika informasi dari data test ikut mempengaruhi proses training. Jika scaler di-fit ke seluruh dataset, mean dan standar deviasi dari data test ikut terbaca. Dengan Pipeline, scaler hanya belajar dari data training pada setiap split atau fold.

### 6.4 Train Model

Training dilakukan di `train_model.py`. Strategi project ini bukan langsung memilih satu model, tetapi membandingkan dua model kandidat:

- Random Forest.
- SVM dengan RBF kernel.

Kedua model dibungkus di dalam `Pipeline` yang berisi `StandardScaler` dan classifier. Pipeline membuat proses training dan inference lebih konsisten karena input akan melalui preprocessing yang sama sebelum diprediksi.

#### Random Forest

Konfigurasi Random Forest:

- `n_estimators=200`.
- `max_depth=None`.
- `min_samples_split=2`.
- `class_weight="balanced"`.
- `random_state=42`.
- `n_jobs=-1`.

Penjelasan dan alasan:

- `n_estimators=200` berarti model memakai 200 decision tree. Semakin banyak tree, prediksi biasanya lebih stabil.
- `max_depth=None` memberi tree kebebasan tumbuh selama masih memenuhi aturan split.
- `class_weight="balanced"` memberi perlindungan jika distribusi kelas tidak seimbang.
- `random_state=42` membuat hasil lebih reproducible.
- `n_jobs=-1` memakai semua core CPU agar training lebih cepat.

Random Forest cocok sebagai baseline kuat karena stabil, mudah dievaluasi, dan bisa memberikan feature importance. Walaupun Random Forest tidak wajib memakai scaling, scaler tetap dimasukkan agar struktur pipeline konsisten dengan SVM.

#### SVM RBF

Konfigurasi SVM:

- `kernel="rbf"`.
- `C=10`.
- `gamma="scale"`.
- `class_weight="balanced"`.
- `probability=False`.
- `random_state=42`.

Penjelasan dan alasan:

- Kernel RBF dipakai karena batas antara gesture pointing dan no pointing bisa bersifat non-linear.
- `C=10` mengatur trade-off antara margin yang lebar dan kesalahan klasifikasi. Nilai ini memberi model fleksibilitas lebih dibanding C yang terlalu kecil.
- `gamma="scale"` membuat nilai gamma dihitung otomatis berdasarkan jumlah fitur dan variansi data.
- `class_weight="balanced"` menjaga model tetap sensitif terhadap kedua kelas.
- `probability=False` membuat inference lebih ringan karena game hanya butuh kelas akhir, bukan probabilitas.

SVM cocok untuk dataset ini karena jumlah fiturnya tidak terlalu banyak, yaitu 20 fitur, dan pola gesture kemungkinan bisa dipisahkan dengan decision boundary non-linear.

#### Cross-Validation

Project menggunakan `StratifiedKFold` dengan 5 fold:

- `n_splits=5`.
- `shuffle=True`.
- `random_state=42`.

Alasannya: cross-validation memberi gambaran performa yang lebih stabil dibanding satu kali train-test split. StratifiedKFold menjaga proporsi kelas tetap seimbang di setiap fold.

#### Model Selection

Model dipilih berdasarkan rata-rata akurasi cross-validation. Dari hasil training yang dijalankan:

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---:|---:|
| Random Forest | 0.9823 | 0.9825 +/- 0.0022 |
| SVM RBF | 0.9864 | 0.9860 +/- 0.0012 |

Karena SVM memiliki rata-rata CV lebih tinggi, model yang tersimpan saat ini adalah pipeline `StandardScaler + SVM`.

Alasannya: pemilihan berdasarkan cross-validation lebih aman daripada hanya berdasarkan test accuracy satu kali. CV menguji model pada beberapa pembagian data, sehingga hasilnya lebih representatif.

### 6.5 Model Evaluation

Evaluasi model dilakukan menggunakan beberapa metrik:

1. Accuracy.

   Accuracy menunjukkan proporsi prediksi yang benar dari seluruh data test.

   Alasannya: karena dataset seimbang, accuracy masih layak digunakan sebagai metrik utama.

2. Precision.

   Precision mengukur seberapa banyak prediksi suatu kelas yang benar-benar sesuai.

   Alasannya: precision penting jika false positive ingin ditekan. Dalam game, false positive bisa membuat paddle aktif padahal gesture sebenarnya tidak aktif.

3. Recall.

   Recall mengukur seberapa banyak data dari suatu kelas yang berhasil dikenali.

   Alasannya: recall penting agar gesture pointing tidak sering gagal dikenali. Jika recall class pointing rendah, paddle akan sering tidak bergerak meskipun pemain sudah menunjuk.

4. F1-score.

   F1-score adalah rata-rata harmonis precision dan recall.

   Alasannya: F1-score berguna untuk melihat keseimbangan antara precision dan recall.

5. Confusion matrix.

   Confusion matrix menunjukkan jumlah prediksi benar dan salah untuk tiap kelas.

   Alasannya: matrix ini memudahkan kita melihat jenis kesalahan model. Misalnya, apakah model lebih sering salah menganggap pointing sebagai no pointing, atau sebaliknya.

6. Feature importance atau model comparison chart.

   Jika Random Forest terpilih, grafik menampilkan fitur paling berpengaruh. Jika SVM terpilih, grafik menampilkan perbandingan performa Random Forest dan SVM.

   Alasannya: visualisasi membantu menjelaskan hasil training secara lebih komunikatif, terutama untuk laporan.

### 6.6 Testing dan Smoke Test

File `test.py` digunakan untuk smoke test. Smoke test bukan training ulang, tetapi pengecekan apakah environment siap menjalankan game.

Hal yang dicek:

- MediaPipe bisa di-import.
- OpenCV bisa di-import.
- Kamera index 0 bisa dibuka.
- NumPy dan scikit-learn tersedia.
- `model/gesture_model.pkl` dan `model/scaler.pkl` bisa diload.
- Input dummy dengan shape `(1, 20)` bisa diprediksi oleh model.
- Pipeline MediaPipe Hands bisa berjalan pada frame kosong.

Alasannya: game real-time melibatkan banyak komponen sekaligus. Jika langsung menjalankan `air_hockey.py` lalu error, sulit mengetahui sumber masalahnya. Smoke test memecah pengecekan menjadi bagian kecil sehingga debugging lebih mudah.

### 6.7 Real-Time Inference dan Game Loop

Tahap akhir project adalah menjalankan `air_hockey.py`. Di tahap ini, model yang sudah dilatih dipakai untuk membaca gesture dari webcam secara real-time.

Alur game loop:

1. Kamera mengambil frame.

   Frame dibaca dari webcam menggunakan OpenCV. Kamera dijalankan melalui thread terpisah agar pengambilan frame tidak terlalu menghambat game loop.

   Alasannya: jika kamera dan rendering berjalan dalam satu alur yang terlalu berat, game bisa terasa lag. Thread kamera membantu frame terbaru selalu tersedia.

2. Frame dibalik horizontal.

   Frame di-flip agar gerakan terasa seperti cermin.

   Alasannya: pengguna lebih mudah mengontrol gerakan jika tangan kanan di dunia nyata juga terlihat bergerak ke kanan di layar.

3. MediaPipe mendeteksi landmark tangan.

   MediaPipe Hands mendeteksi maksimal 2 tangan. Setiap tangan menghasilkan 21 landmark.

   Alasannya: game ini mendukung 2 pemain lokal, sehingga sistem perlu bisa membaca dua tangan dalam satu frame.

4. Landmark diubah menjadi fitur jarak.

   Fungsi `landmarks_to_distances()` mengambil koordinat 21 landmark, menghitung jarak masing-masing landmark ke wrist, lalu menormalisasi jarak dengan nilai maksimum. Setelah itu `dist_0` dibuang sehingga tersisa 20 fitur.

   Alasannya: format fitur real-time harus sama dengan format fitur saat training. Jika training memakai 20 jarak landmark dan inference memakai fitur lain, model akan menerima input yang tidak sesuai.

5. Model memprediksi gesture.

   Model menghasilkan:

   - `0`: no pointing atau closed fist, paddle tidak bergerak.
   - `1`: pointing, paddle mengikuti ujung jari telunjuk.

   Alasannya: gesture classifier menjadi gerbang kontrol. Paddle hanya aktif ketika gesture dianggap valid.

6. Inference di-throttle.

   Prediksi gesture dilakukan setiap 2 frame, bukan setiap frame.

   Alasannya: gesture tangan tidak berubah drastis dalam selang waktu sangat pendek. Dengan memakai hasil frame sebelumnya untuk frame genap, CPU usage bisa turun tanpa membuat kontrol terasa jauh lebih lambat.

7. Posisi ujung jari dipakai sebagai target paddle.

   Landmark 8 dari MediaPipe adalah ujung jari telunjuk. Koordinat landmark ini dipetakan ke resolusi layar game.

   Alasannya: landmark ujung telunjuk adalah titik paling natural untuk dijadikan kontrol karena pemain memang diminta menunjuk.

8. Player ditentukan dari posisi tangan.

   Jika posisi tangan berada di kiri layar, sistem menganggapnya Player 1. Jika berada di kanan layar, sistem menganggapnya Player 2.

   Alasannya: cara ini sederhana dan cocok untuk game dua pemain lokal tanpa perlu proses identifikasi pemain yang lebih kompleks.

9. Paddle dibatasi area geraknya.

   Player 1 hanya bisa bergerak di setengah kiri layar, sedangkan Player 2 hanya bisa bergerak di setengah kanan layar.

   Alasannya: aturan ini mengikuti air hockey asli, yaitu setiap pemain punya wilayah masing-masing. Ini juga mencegah satu pemain mengganggu area lawan secara langsung.

10. Puck physics dijalankan.

   Puck memiliki posisi, velocity, pantulan dinding, tabrakan dengan paddle, speed boost, speed cap, dan goal detection.

   Alasannya: physics membuat game terasa responsif dan punya tantangan. Tanpa physics, game hanya menjadi visualisasi posisi tangan.

11. Rendering AR dilakukan.

   Background kamera digelapkan, lalu table overlay, puck, paddle, score, efek bloom, goal flash, dan screen shake digambar di atasnya.

   Alasannya: overlay ini menciptakan kesan augmented reality. Pemain tetap melihat dunia nyata dari kamera, tetapi elemen game muncul di atasnya.

12. Game selesai saat skor mencapai 7.

   Pemain pertama yang mencapai `WIN_SCORE = 7` dinyatakan menang.

   Alasannya: batas skor memberi tujuan yang jelas dan membuat game punya kondisi selesai.

## 7. Konfigurasi Penting

### 7.1 Konfigurasi Game

| Konfigurasi | Nilai | Alasan |
|---|---:|---|
| `DW, DH` | 1280, 720 | Resolusi 16:9 yang cukup besar untuk dua pemain |
| `CX` | 640 | Garis tengah untuk membagi area Player 1 dan Player 2 |
| `GOAL_Y1, GOAL_Y2` | 230, 490 | Membuat area goal berada di tengah secara vertikal |
| `MALLET_R` | 40 | Ukuran paddle cukup besar untuk dikontrol tangan |
| `PUCK_R` | 24 | Ukuran puck terlihat jelas tetapi tidak terlalu besar |
| `WIN_SCORE` | 7 | Skor akhir yang cukup singkat untuk demo |
| `CAM_ZOOM` | 1.6 | Memberi crop/zoom agar area bermain terasa lebih fokus |
| `Puck.BASE_SPEED` | 18.0 | Kecepatan awal puck |
| `Puck.MAX_SPEED` | 64.0 | Batas kecepatan agar puck tidak terlalu sulit dilihat |
| `Puck.HIT_BOOST` | 1.07 | Membuat rally makin cepat setelah beberapa hit |
| `Mallet.lerp` | 0.38 | Menghaluskan gerakan paddle agar tidak jitter |
| `min_detection_confidence` | 0.7 | Deteksi tangan lebih selektif |
| `min_tracking_confidence` | 0.5 | Tracking tetap cukup toleran saat tangan bergerak |

### 7.2 Konfigurasi Training

| Konfigurasi | Nilai | Alasan |
|---|---:|---|
| `test_size` | 0.2 | 20 persen data untuk evaluasi akhir |
| `random_state` | 42 | Hasil eksperimen lebih mudah direproduksi |
| `n_splits` | 5 | Cross-validation cukup stabil tanpa terlalu mahal |
| RF `n_estimators` | 200 | Prediksi Random Forest lebih stabil |
| RF `class_weight` | balanced | Perlindungan terhadap imbalance |
| SVM `kernel` | rbf | Mampu menangkap pola non-linear |
| SVM `C` | 10 | Memberi fleksibilitas pada decision boundary |
| SVM `gamma` | scale | Gamma dihitung otomatis dari data |

### 7.3 Konfigurasi Data Preparation

| Konfigurasi | Nilai | Alasan |
|---|---:|---|
| `header` | None | Dataset mentah tidak memiliki header |
| `on_bad_lines` | skip | Baris rusak dilewati agar pipeline tetap berjalan |
| Kolom dihapus | `dist_0` | Fitur selalu 0 dan tidak informatif |
| Missing handling | `dropna()` | Model tidak menerima nilai kosong |
| Outlier threshold | > 2.0 | Deteksi awal nilai fitur yang tidak wajar |

## 8. Artefak Output

Setelah pipeline dijalankan, project menghasilkan beberapa artefak:

| Artefak | Sumber | Fungsi |
|---|---|---|
| `data/gesture_data.csv` | `prepare_dataset.py` | Dataset bersih untuk training |
| `model/gesture_model.pkl` | `train_model.py` | Pipeline model terbaik untuk inference |
| `model/scaler.pkl` | `train_model.py` | Scaler terpisah untuk kompatibilitas atau testing |
| `model/confusion_matrix.png` | `train_model.py` | Visualisasi evaluasi prediksi model |
| `model/feature_importance.png` | `train_model.py` | Visualisasi feature importance atau perbandingan model |
| `model/training_curve.png` | File model lama / artefak existing | Grafik tambahan yang masih ada di folder model, tetapi bukan output utama kode `train_model.py` saat ini |

Catatan penting: dokumentasi README lama masih menyebut MLP dan `training_curve.png` sebagai output utama. Namun implementasi `train_model.py` saat ini membandingkan Random Forest dan SVM, lalu menyimpan model terbaik. Berdasarkan model yang tersimpan sekarang, pipeline aktif adalah `StandardScaler + SVM`.

## 9. Cara Menjalankan Project

1. Install dependency.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Siapkan dataset.

Pastikan file dataset mentah berada di:

```text
data/hand-gestures.csv
```

3. Jalankan data preparation.

```bash
python prepare_dataset.py
```

Output:

```text
data/gesture_data.csv
```

4. Jalankan training.

```bash
python train_model.py
```

Output utama:

```text
model/gesture_model.pkl
model/scaler.pkl
model/confusion_matrix.png
model/feature_importance.png
```

5. Jalankan smoke test.

```bash
python test.py
```

Tujuannya untuk memastikan dependency, kamera, model, dan MediaPipe pipeline sudah siap.

6. Jalankan game.

```bash
python air_hockey.py
```

Kontrol:

- Angkat jari telunjuk untuk menggerakkan paddle.
- Kepalkan tangan atau tidak pointing untuk membuat paddle diam.
- Tekan `Q` untuk keluar.
- Tekan `R` untuk restart.

## 10. Ringkasan Pipeline Final

Project ini dapat dipahami sebagai rangkaian proses berikut:

1. Dataset gesture mentah dikumpulkan dalam bentuk CSV.
2. Dataset dibersihkan, kolom diberi nama, fitur tidak informatif dihapus, missing value dibuang, dan dataset bersih disimpan.
3. Dataset dianalisis secara singkat untuk mengecek jumlah data, class balance, rentang fitur, dan missing value.
4. Data dipisah menjadi fitur dan target.
5. Data dibagi menjadi train dan test dengan stratifikasi.
6. Random Forest dan SVM dilatih di dalam Pipeline yang berisi StandardScaler.
7. Kedua model dievaluasi memakai test accuracy, classification report, confusion matrix, dan 5-fold cross-validation.
8. Model terbaik dipilih berdasarkan rata-rata cross-validation.
9. Model dan scaler disimpan sebagai file `.pkl`.
10. Smoke test memastikan environment dan model siap digunakan.
11. Game mengambil frame webcam secara real-time.
12. MediaPipe mendeteksi landmark tangan.
13. Landmark diubah menjadi 20 fitur jarak yang formatnya sama seperti data training.
14. Model memprediksi gesture.
15. Gesture pointing mengaktifkan paddle, sedangkan no pointing membuat paddle diam.
16. Game physics mengatur puck, tabrakan, goal, skor, dan kemenangan.
17. OpenCV menggambar semua elemen game di atas feed kamera sebagai AR overlay.

## 11. Kesimpulan

AR Air Hockey adalah project Computer Vision yang lengkap karena mencakup seluruh alur dari dataset sampai aplikasi real-time. Project ini tidak hanya mendeteksi tangan, tetapi juga mengubah hasil deteksi menjadi fitur numerik, melatih model gesture classifier, mengevaluasi model, lalu memakai model tersebut untuk mengontrol game.

Kekuatan utama project ini ada pada konsistensi pipeline. Fitur yang digunakan saat training sama dengan fitur yang dibuat saat inference, yaitu 20 jarak Euclidean landmark tangan terhadap wrist yang sudah dinormalisasi. Selain itu, penggunaan Pipeline scikit-learn membantu mencegah data leakage dan membuat proses training lebih rapi.

Berdasarkan hasil evaluasi, SVM RBF menjadi model terbaik dengan akurasi test sekitar 98.64 persen dan rata-rata 5-fold cross-validation sekitar 98.60 persen. Dengan performa tersebut, model cukup kuat untuk digunakan sebagai kontrol gesture dalam game AR Air Hockey secara real-time.
