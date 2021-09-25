# [PerCom 2022 Submission] Weight Separation for Memory-Efficient and Accurate Deep Multitask Learning on Embedded Systems

## Introduction
This is an anonymous open-source repository of the [PerCom 2022](https://www.percom.org/) submission titled "***Weight Separation for Memory-Efficient and Accurate Deep Multitask Learning***". It enables extreme low-memory and accurate deep multitask learning with two types of weight parameters applied to two levels of the system memory hierarchy, i.e., the shared weights for primary memory (level-1) and the exclusive weight for secondary memory (level-2). It first groups the weight parameters of DNNs (deep neural networks) into a set of memory blocks we call weight-pages and then optimally distributes them to primary and secondary memory in order to achieve two seemingly incompatible objectives at the same time, i.e., 1) memory reduction of weight parameters and 2) performance (prediction accuracy) guarantee.

This repository implements 1) *weight separation* of multiple heterogeneous DNNs of arbitrary network architectures and 2) *multitask DNN execution*. For the reviewers' convenience, we provide a step-by-step guideline here for the showcase of the weight separation and the execution of the ten DNNs that are used for the multitask IoT device, one of the systems we implement in the paper. The sizes of weight parameters of those DNNs are small (up to 290KB), so the entire process of weight separation can be easily demonstrated in a reasonable time without requiring us to spend several days. The ten DNNs (and datasets) consist of state-of-the-art image, audio, and sensor-based tasks, i.e., 1) [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 2) [MNIST](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), 3) [Fashion MNIST (FMNIST)](https://arxiv.org/pdf/1708.07747.pdf), 4) [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf), 5) [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf), 6) [Obstacle Avoidance (OBS)](https://github.com/varunverlencar/Dynamic-Obstacle-Avoidance-DL), 7) [GoogleSpeechCommands V2(GSC)](https://arxiv.org/abs/1804.03209), 8) [Environmental Sound Classification (ESC-10)](https://dl.acm.org/doi/pdf/10.1145/2733373.2806390), 9) [UrbanSound 8K (US8K)](https://dl.acm.org/doi/pdf/10.1145/2647868.2655045), and 10) [Heterogeneous Human Activity Recognition (HHAR)](https://dl.acm.org/doi/pdf/10.1145/2809695.2809718). Our experiment result shows that the weight separation packs a total of ***2,682KB weight parameters of the ten DNNs into 554KB (5.9x)*** with comparable prediction accuracy (***i.e., -0.22% on average***) and improves the weight load time (DNN switching time) by ***52x***.

## Software Install and Setup
The weight separation is implemented by using Python, TensorFlow, and NVIDIA CUDA (custom TensorFlow operation). The TensorFlow version should be lower than or equal to 1.13.2; the latest version (2.0) seems to have a problem of executing custom operations. We used Python 2.7, Tensorflow 1.13.1, and CUDA 10.0. A GPU is required to perform the weight separation, i.e., weight-page formation, separation, overlapping, and optimization, as well as the multitak DNN execution. We used an NVIDIA RTX 20280 Ti GPU.

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow (<= 1.13.2)](https://www.tensorflow.org/).

**Step 3.** Install [NVIDIA CUDA (>= 10.0)](https://developer.nvidia.com/cuda-downloads).

**Step 4.** Clone this WeightSeparation repository.
```sh
$ git clone https://github.com/twolevelasplos2021/WeightSeparation.git
Cloning into 'WeightSeparation'...
remote: Enumerating objects: 124, done.
remote: Counting objects: 100% (124/124), done.
remote: Compressing objects: 100% (83/83), done.
remote: Total 124 (delta 55), reused 96 (delta 38), pack-reused 0
Receiving objects: 100% (124/124), 7.48 MiB | 45.32 MiB/s, done.
Resolving deltas: 100% (55/55), done.
```

## 1) Download Datasets (Preliminary Step 1)
Before getting into the weight separation, download the ten datasets (except the MNIST dataset) by executing the following script (*download_dataset.sh*). The script uses [curl](https://curl.haxx.se/download.html) for downloading. 
```sh
$ ./download_dataset.sh 
[1/9] Downloading the cifar10 dataset...
Downloading cifar10/cifar10_test_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2503      0 --:--:-- --:--:-- --:--:--  2503
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  234M    0  234M    0     0  77.3M      0 --:--:--  0:00:03 --:--:--  111M
Downloading cifar10/cifar10_test_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0    980      0 --:--:-- --:--:-- --:--:--   980
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  781k  100  781k    0     0   954k      0 --:--:-- --:--:-- --:--:--  954k
Downloading cifar10/cifar10_train_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2935      0 --:--:-- --:--:-- --:--:--  2935
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 1171M    0 1171M    0     0   103M      0 --:--:--  0:00:11 --:--:--  111M
Downloading cifar10/cifar10_train_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0    910      0 --:--:-- --:--:-- --:--:--   908
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 3906k    0 3906k    0     0  4449k      0 --:--:-- --:--:-- --:--:-- 4449k

...
...
...

[9/9] Downloading the us8k dataset...
Downloading us8k/US8K_test_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   3044      0 --:--:-- --:--:-- --:--:--  3044
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  202M    0  202M    0     0  80.7M      0 --:--:--  0:00:02 --:--:--  111M
Downloading us8k/US8K_test_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2113      0 --:--:-- --:--:-- --:--:--  2113
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  422k  100  422k    0     0   705k      0 --:--:-- --:--:-- --:--:--  705k
Downloading us8k/US8K_train_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2443      0 --:--:-- --:--:-- --:--:--  2428
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 1826M    0 1826M    0     0  94.4M      0 --:--:--  0:00:19 --:--:-- 99.6M
Downloading us8k/US8K_train_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   1651      0 --:--:-- --:--:-- --:--:--  1651
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 3801k    0 3801k    0     0  5445k      0 --:--:-- --:--:-- --:--:-- 5445k
```

## 2) Prepare and Train DNN Models (Preliminary Step 2)
The next preliminary step is to obtain and train DNN models for the ten datasets. For the reviewers' convenience, we include pre-trained models of the ten DNNs in this repository. They are located in the following folders.
```sh
$ ls -d cifar10 mnist fmnist gtsrb  svhn obs gsc esc10 us8k hhar
cifar10  esc10  fmnist  gsc  gtsrb  hhar  mnist  obs  svhn  us8k 
```

The number of weight parameters, memory usage, and prediction accuracy of each DNN model are shown in the below table. In short, a total of ***667,582 weight parameters (2,682KB)*** are packedinto ***113,664 weight parameters (554KB)***, achieving ***5.9x*** of memory packing efficiency ***without modifying the network arechitectures of DNNs (i.e., not compression or pruning)***.

| DNN | Number of weights | Memory (KB) | Prediction Accuracy (%) |
| :-------------: | -------------: | -------------: | -------------: |
| CIFAR-10 | 64,670 | 260 | 71.05 |
| MNIST  | 65,402 | 262 | 98.61 |
| FMNIST  | 69,966 | 280 | 85.80 |
| GTSRB | 68,639 | 276 | 94.33 |
| SVHN | 64,670 | 260 | 88.75 |
| OBS | 72,012 | 288 | 99.96 |
| GSC | 65,101 | 262 | 73.02 |
| ESC-10 | 65,154 | 262 | 80.62 |
| US8K | 66,854 | 268 | 42.90 |
| HHAR | 65,114 | 262 | 88.85 |
| **Total** | ***667,582*** | ***2,682*** | - |
| **Weight-Separated** | ***113,664*** | ***554*** | - |

## 3) Step 1: Weight-Page Formation, Separation, and Overlapping
The first step of weight separation is the weight-page formation, separation, and overlapping, which is performed by a Python script (*WeightSeparation.py.py*). It 1) forms weight-pages from the DNNs, 2) computes Fisher information of the weight-pages, 3) separate them into the sharable and exclusive weight-pages, and 4) overlaps the sharable weight-pages onto the shared weight-pages, as described in the paper.

Perform the weight-page formation, separation, and overlapping of the ten DNNs with the following shell script (*formation.sh*).
```sh
$ ./formation.sh 
init new weight pages
no exclusive weight pages
add_vnn
cifar10/cifar10_network_weight.npy
compute_fisher
do_compute_fisher
  0, data_idx:  2664
  1, data_idx: 16646
  2, data_idx: 19677
  3, data_idx: 13262
  4, data_idx: 23087
  5, data_idx: 23571
  6, data_idx: 19225
  7, data_idx: 15170
  8, data_idx: 37487
  9, data_idx: 27512
 10, data_idx: 40564
 11, data_idx: 37944
 12, data_idx: 26872
 13, data_idx: 27129
 14, data_idx: 38409
 15, data_idx: 18392
 16, data_idx:  8548
 17, data_idx: 14439
 18, data_idx: 49649
 19, data_idx:   890
 20, data_idx: 27843
 21, data_idx: 22480
 22, data_idx: 49940
 23, data_idx: 20306
 24, data_idx: 46295
 25, data_idx:  1783
 26, data_idx: 45334
 27, data_idx:   766
 28, data_idx:  7136
 29, data_idx: 15926
 30, data_idx: 39952
 31, data_idx: 10561
 32, data_idx: 44067
 33, data_idx: 43028
 34, data_idx: 30950
 35, data_idx:    90
 36, data_idx: 13411
 37, data_idx:  4917
 38, data_idx: 47982
 39, data_idx: 25341
 40, data_idx: 29989
 41, data_idx: 16357
 42, data_idx: 28350
 43, data_idx: 29327
 44, data_idx:  4842
 45, data_idx: 48304
 46, data_idx:  8721
 47, data_idx: 21059
 48, data_idx:  8157
 49, data_idx: 19140
cifar10/cifar10_network_fisher.npy
compute_loss
0 30000
cifar10 inference accuracy: 0.776800
30000 60000
cifar10 inference accuracy: 0.772950
exclusive_weight_page_list [0, 9]
num_of_weight 64670
num_of_weight_page 127
num_of_exclusive_weight 1024
num_of_exclusive_weight_page 2
num_of_inexclusive_weight 63646
num_of_inexclusive_weight_page 125
assign_page 0.006 ms
inexclusive_weight_page_list
[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124]
vnn.weight_page_list [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124]
[calculate_cost]
toal_cost: 0.0
125 pages allocated for 63646 weights
total_network_cost: 0

...
...
...

inexclusive_weight_page_list
[114   7  50  37  13   1 126  61 117  69   5 102  76  46  74   2  49  24
 119 113  43  51  59   8  30 127  54   9  25  40  27  80 125   0  56  86
  52  39  81 122  47  48  62  77  85  82  72  65 103  95  10  64  35  68
 100  45  16  44  42  75  63  58  21  17  33 124  34  22  73  14  57   6
  41  32  11   3  31  83  18  71  78  90 111 120   4  23  87 115  98  15
 105  66  92 101  99 112 121  60  55 110 118  19  53  26  20  96  84 116
  29  36 109 123  12  70  93  91  89  94 104  38 106  97  88 108 107  79
  67]
vnn.weight_page_list [114   7  50  37  13   1 126  61 117  69   5 102  76  46  74   2  49  24
 119 113  43  51  59   8  30 127  54   9  25  40  27  80 125   0  56  86
  52  39  81 122  47  48  62  77  85  82  72  65 103  95  10  64  35  68
 100  45  16  44  42  75  63  58  21  17  33 124  34  22  73  14  57   6
  41  32  11   3  31  83  18  71  78  90 111 120   4  23  87 115  98  15
 105  66  92 101  99 112 121  60  55 110 118  19  53  26  20  96  84 116
  29  36 109 123  12  70  93  91  89  94 104  38 106  97  88 108 107  79
  67]
[calculate_cost]
toal_cost: 17.075286126404535
127 pages allocated for 64602 weights
total_network_cost: 114.54198178648949
get_overlapping_loss
v_train
step 0, training accuracy: 0.080000 original loss: 7.806100 overlapping loss: 0.004187
step 0, Validation accuracy: 0.116513
step 100, training accuracy: 0.730000 original loss: 5.477027 overlapping loss: 0.003609
step 100, Validation accuracy: 0.404023
get new weight for 0.40402347
step 200, training accuracy: 0.730000 original loss: 5.310961 overlapping loss: 0.003668
step 200, Validation accuracy: 0.709137
get new weight for 0.7091366
step 300, training accuracy: 0.810000 original loss: 5.310233 overlapping loss: 0.003750
step 300, Validation accuracy: 0.772003
get new weight for 0.77200335
step 400, training accuracy: 0.850000 original loss: 5.089818 overlapping loss: 0.003837
step 400, Validation accuracy: 0.797150
get new weight for 0.79715
step 500, training accuracy: 0.890000 original loss: 4.997283 overlapping loss: 0.003909
step 500, Validation accuracy: 0.857502
get new weight for 0.8575021
step 600, training accuracy: 0.890000 original loss: 4.932966 overlapping loss: 0.003977
step 600, Validation accuracy: 0.828164
step 700, training accuracy: 0.920000 original loss: 4.870730 overlapping loss: 0.004038
step 700, Validation accuracy: 0.864208
get new weight for 0.86420786
step 800, training accuracy: 0.960000 original loss: 4.873968 overlapping loss: 0.004096
step 800, Validation accuracy: 0.880972
get new weight for 0.8809723
step 900, training accuracy: 0.920000 original loss: 4.881790 overlapping loss: 0.004139
step 900, Validation accuracy: 0.839899
step 1000, training accuracy: 0.970000 original loss: 4.797315 overlapping loss: 0.004184
step 1000, Validation accuracy: 0.865046
step 1100, training accuracy: 0.950000 original loss: 4.802908 overlapping loss: 0.004222
step 1100, Validation accuracy: 0.870914
step 1200, training accuracy: 0.900000 original loss: 4.890184 overlapping loss: 0.004259
step 1200, Validation accuracy: 0.873428
step 1300, training accuracy: 0.960000 original loss: 4.810242 overlapping loss: 0.004292
step 1300, Validation accuracy: 0.859179
step 1400, training accuracy: 0.940000 original loss: 4.836988 overlapping loss: 0.004323
step 1400, Validation accuracy: 0.880134
step 1500, training accuracy: 0.950000 original loss: 4.837905 overlapping loss: 0.004354
step 1500, Validation accuracy: 0.883487
get new weight for 0.883487
step 1600, training accuracy: 0.950000 original loss: 4.788506 overlapping loss: 0.004386
step 1600, Validation accuracy: 0.886002
get new weight for 0.8860017
step 1700, training accuracy: 0.970000 original loss: 4.789769 overlapping loss: 0.004412
step 1700, Validation accuracy: 0.861693
step 1800, training accuracy: 0.910000 original loss: 4.839781 overlapping loss: 0.004447
step 1800, Validation accuracy: 0.900251
get new weight for 0.90025145
step 1900, training accuracy: 0.950000 original loss: 4.797198 overlapping loss: 0.004477
step 1900, Validation accuracy: 0.883487
step 1999, training accuracy: 0.980000 original loss: 4.748354 overlapping loss: 0.004511
step 1999, Validation accuracy: 0.884325
hhar/hhar_weight.npy
```

## 4) Step 2: Weight-Page Optimization
The next step of weight separation is the weight-page optimization, which combines the overlapped weight-pages into unified shared weight-pages and optimizes them for the ten DNN tasks, along with the exclusive weight-pages. Here, we perform the optimization by executing a shell script (*optimization.sh*).
```sh
$ ./optimization.sh 
1-th optimization
get_overlapping_loss
v_train
step 0, training accuracy: 0.150000 original loss: 7.230406 overlapping loss: 0.004216
step 0, Validation accuracy: 0.172200
step 31, training accuracy: 0.290000 original loss: 6.274111 overlapping loss: 0.003656
step 31, Validation accuracy: 0.412500
get new weight for 0.4125
cifar10/cifar10_weight.npy
get_overlapping_loss
v_train
Extracting /.../WeightSeparation/mnist/MNIST_data/train-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/t10k-labels-idx1-ubyte.gz
step 0, training accuracy: 0.290000 original loss: 16.312901 overlapping loss: 0.002309
step 0, Validation accuracy: 0.339000
step 31, training accuracy: 0.870000 original loss: 5.326897 overlapping loss: 0.002031
step 31, Validation accuracy: 0.897700
get new weight for 0.8977
mnist/mnist_weight.npy
get_overlapping_loss
v_train
step 0, training accuracy: 0.240000 original loss: 15.012118 overlapping loss: 0.002933
step 0, Validation accuracy: 0.216400
step 31, training accuracy: 0.750000 original loss: 5.590587 overlapping loss: 0.002568
step 31, Validation accuracy: 0.703700
get new weight for 0.7037
fmnist/fmnist_weight.npy
get_overlapping_loss
v_train
step 0, training accuracy: 0.140000 original loss: 20.653532 overlapping loss: 0.002693
step 0, Validation accuracy: 0.137767
step 31, training accuracy: 0.310000 original loss: 7.459014 overlapping loss: 0.002351
step 31, Validation accuracy: 0.297466
get new weight for 0.29746634
gtsrb/gtsrb_weight.npy
get_overlapping_loss
v_train
step 0, training accuracy: 0.790000 original loss: 5.526350 matching loss: 0.000325
step 0, Validation accuracy: 0.758067
step 31, training accuracy: 0.850000 original loss: 5.259809 matching loss: 0.000284
step 31, Validation accuracy: 0.855716
get new weight for 0.85571605
svhn/svhn_weight.npy

...
...
...

100-th evaluation
cifar10 inference accuracy: 0.699000
Extracting /.../WeightSeparation/mnist/MNIST_data/train-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/t10k-labels-idx1-ubyte.gz
mnist inference accuracy: 0.984500
fmnist inference accuracy: 0.873500
gtsrb inference accuracy: 0.943400
svhn inference accuracy: 0.876191
obstacle inference accuracy: 0.997898
gsc inference accuracy: 0.741109
esc10 inference accuracy: 0.794929
US8K inference accuracy: 0.428300
hhar inference accuracy: 0.878300
```

After the optimization is completed, the shared and exclusive weight-pages are generated (*shared_weight_page.npy* and *exclusive_weight_page.npy*). The final prediction accuracy of each DNN can be checked with the following Python script (*WeightSeparation.py*).
```sh
$ python WeightSeparation.py -mode=e -vnn_name=cifar10
cifar10 inference accuracy: 0.699000
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=mnist
Extracting /.../WeightSeparation/mnist/MNIST_data/mnist/MNIST_data/train-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/mnist/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/mnist/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /.../WeightSeparation/mnist/MNIST_data/mnist/MNIST_data/t10k-labels-idx1-ubyte.gz
mnist inference accuracy: 0.984500
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=fmnist
fmnist inference accuracy: 0.873500
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=gtsrb
gtsrb inference accuracy: 0.943400
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=svhn
svhn inference accuracy: 0.876191
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=obs
obstacle inference accuracy: 0.997898
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=obs
gsc inference accuracy: 0.741109
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=esc10
esc10 inference accuracy: 0.794929
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=us8k
US8K inference accuracy: 0.428300
```
```sh
$ python WeightSeparation.py -mode=e -vnn_name=hhar
hhar inference accuracy: 0.878300
```

The below table is the comparison of prediction accuracy between the baseline DNNs (before) and the weight-separated DNNs (after). The weight-separated DNNs achieve comparable prediction accuracy to the baseline DNNs, i.e., -0.22% on average.

| | Prediction Accuracy (%) | Prediction Accuracy (%) | Difference (%)
| :-------------: | -------------: | -------------: | ------------: |
| DNN | Non-Separated DNN | ***Weight-Separated DNN*** | |
| CIFAR-10 | 71.05 | ***69.90*** | -1.15 |
| MNIST  | 98.61 | ***98.45*** |  -0.16 |
| FMNIST  | 85.80 | ***87.35*** | +1.55 |
| GTSRB | 94.33 | ***94.34*** | +0.01 |
| SVHN | 88.75 | ***87.61*** | -1.14 |
| OBS | 99.96 | ***99.78*** | -0.18 |
| GSC | 73.02 | ***74.11*** | +1.09 |
| ESC-10 | 80.62 | ***79.49*** | -1.13 |
| US8K | 42.90 | ***42.83*** | -0.07 |
| HHAR | 88.85 | ***87.83*** | -1.02 |

## 5) Multitask DNN Execution with Weight Separation (ours) vs. without Weight Separation
Once the weight separation is completed, the shared and exclusive weight-pages are generated (*shared_weight_page.npy* and *exclusive_weight_page.npy*) and loaded into the GPU RAM and CPU RAM, respectively. With the weight separation, the multitask DNNs can exploit the low access latency of GPU RAM, which enables fast and responsive execution while ensuring the prediction accuracy.

We compare the DNN switching (weight parameter loading) and the execution time of the weight-separated DNNs against the non-separated baseline DNNs.

First, the weight-separated DNN execution is performed by the following Python script (i.e., *ws_execution.py*) that executes 50 random DNNs and measures the DNN switching and execution time. The result shows that the total DNN switching time and execution time of the 50 DNN execution is ***39.942 ms*** and ***12705.812 ms***, respectively.

```sh
$ python ws_execution.py
shared_weight GPU address: 0x7F2B05000000
init shared_weight 85.348 ms
[VNN 0][cifar10] init page table 3.494 ms
[VNN 1][esc10] init page table 2.842 ms
[VNN 2][fmnist] init page table 2.403 ms
[VNN 3][gsc] init page table 2.533 ms
[VNN 4][gtsrb] init page table 2.542 ms
[VNN 5][hhar] init page table 2.638 ms
[VNN 6][mnist] init page table 2.665 ms
[VNN 7][obs] init page table 2.716 ms
[VNN 8][svhn] init page table 2.877 ms
[VNN 9][us8k] init page table 2.770 ms
[0/50] total_weight_load_time: 0.000 ms, total_execution_time: 0.000 ms
[Executing] esc10
weights load time : 0.782 ms
esc10 inference accuracy: 0.794929
DNN execution time: 836.526 ms
[1/50] total_weight_load_time: 0.782 ms, total_execution_time: 836.526 ms
[Executing] esc10
weights load time : 0.771 ms
esc10 inference accuracy: 0.794929
DNN execution time: 98.757 ms
[2/50] total_weight_load_time: 1.553 ms, total_execution_time: 935.283 ms
[Executing] gsc
weights load time : 0.426 ms
gsc inference accuracy: 0.741109
DNN execution time: 288.221 ms
[3/50] total_weight_load_time: 1.979 ms, total_execution_time: 1223.504 ms
[Executing] hhar
weights load time : 0.269 ms
hhar inference accuracy: 0.878300
DNN execution time: 163.167 ms

...
...
...

[47/50] total_weight_load_time: 38.851 ms, total_execution_time: 12650.815 ms
[Executing] hhar
weights load time : 0.281 ms
hhar inference accuracy: 0.878300
DNN execution time: 71.347 ms
[48/50] total_weight_load_time: 39.132 ms, total_execution_time: 12722.162 ms
[Executing] gsc
weights load time : 0.337 ms
gsc inference accuracy: 0.741109
DNN execution time: 156.140 ms
[49/50] total_weight_load_time: 39.469 ms, total_execution_time: 12878.302 ms
[Executing] obs
weights load time : 1.068 ms
obstacle inference accuracy: 0.997898
DNN execution time: 168.728 ms
total weights load time : 39.942 ms
total DNN execution time: 12705.812 ms
```

Next, the non-separated execution is performed by the following Python script (i.e., *baseline_execution.py*) that executes 50 random DNNs and measures the DNN switching and execution time. The result shows that the total DNN switching time and execution time of the 50 DNN execution is ***2081.574 ms*** and ***12874.752 ms***, respectively.
```sh
$ python baseline_execution.py
[0/50] total_weight_load_time: 0.000 ms, total_execution_time: 0.000 ms
[Executing] us8k
weights load time : 41.432 ms
US8K inference accuracy: 0.439408
DNN execution time: 1150.419 ms
[1/50] total_weight_load_time: 41.432 ms, total_execution_time: 1150.419 ms
[Executing] esc10
weights load time : 43.637 ms
esc10 inference accuracy: 0.816239
DNN execution time: 126.486 ms
[2/50] total_weight_load_time: 85.069 ms, total_execution_time: 1276.905 ms
[Executing] hhar
weights load time : 26.576 ms
hhar inference accuracy: 0.898575
DNN execution time: 155.811 ms
[3/50] total_weight_load_time: 111.645 ms, total_execution_time: 1432.716 ms
[Executing] hhar
weights load time : 27.773 ms
hhar inference accuracy: 0.898575
DNN execution time: 70.395 ms
[4/50] total_weight_load_time: 139.418 ms, total_execution_time: 1503.111 ms
[Executing] hhar
weights load time : 26.429 ms
hhar inference accuracy: 0.898575
DNN execution time: 69.014 ms

...
...
...

[47/50] total_weight_load_time: 1761.014 ms, total_execution_time: 10700.530 ms
[Executing] hhar
weights load time : 29.869 ms
hhar inference accuracy: 0.898575
DNN execution time: 72.897 ms
[48/50] total_weight_load_time: 1790.883 ms, total_execution_time: 10773.427 ms
[Executing] gsc
weights load time : 40.921 ms
gsc inference accuracy: 0.742026
DNN execution time: 146.362 ms
[49/50] total_weight_load_time: 1831.804 ms, total_execution_time: 10919.789 ms
[Executing] us8k
weights load time : 27.650 ms
US8K inference accuracy: 0.439408
DNN execution time: 229.625 ms
total weights load time : 2081.574 ms
total DNN execution time: 12874.752 ms
```

It shows that the weight-separated DNN execution accelerates the DNN switching time by ***52x (39.942 ms vs. 2081.574 ms)***.

| | DNN Switching Time (ms) | DNN Execution Time (ms)
| :-------------: | -------------: | -------------: |
| ***Weight-Separated Execution*** | ***39.942*** | ***12705.812*** |
| Non-Seperated Execution | 2081.574 | 12874.752 |
