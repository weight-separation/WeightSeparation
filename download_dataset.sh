#!/bin/bash

download_file()
{
	fileid=$1
	filename=$2
	echo "Downloading ${filename}"
	curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
	curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
}

dnn_name="cifar10"
echo "[1/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1y92NkPwfKGlH6QvaRd1AH5HsuXxS_VTq" "${dnn_name}/cifar10_test_data.npy"
download_file "1nIEqxa4HPsXfOxka0X1YTHPtIQTd3gI0" "${dnn_name}/cifar10_test_label.npy"
download_file "1GiwtMZWQLGuTCyuuc42lutD2QI0ACDke" "${dnn_name}/cifar10_train_data.npy"
download_file "1NBYkQf50ty3EzQiF1SKhAztUSiwejfxA" "${dnn_name}/cifar10_train_label.npy"

dnn_name="esc10"
echo "[2/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1MLEzjpPo2aDDl3zz1mJIttiu9Leztxls" "${dnn_name}/esc10_test_data.npy"
download_file "1HYp1T4_iPgmxY1Cc5Mj3pf-PFivNVTwb" "${dnn_name}/esc10_test_label.npy"
download_file "1KOAykNhNEkP8q8Ib-VF52w_gHgudn2IS" "${dnn_name}/esc10_train_data.npy"
download_file "1xx3hljerqLm50I9mr7B1EBAAPVpHcN9D" "${dnn_name}/esc10_train_label.npy"

dnn_name="fmnist"
echo "[3/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1xIcFXcu9KsTOFpdRRrHfdPG3SniCdhhb" "${dnn_name}/fmnist_test_data.npy"
download_file "1K8SLjM7PeC0XIjNetJ8q4ZaDs3Se5T99" "${dnn_name}/fmnist_test_label.npy"
download_file "1tITHwmRwkTAGMCND5I4LIfxgy3Gn33xs" "${dnn_name}/fmnist_train_data.npy"
download_file "1mtTjU1ib2whkleH2sgrUx7sMlgEz3suL" "${dnn_name}/fmnist_train_label.npy"

dnn_name="gsc"
echo "[4/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "10xHTi2jJyWfhzJwZhwrpiJ1WNXWuSx5C" "${dnn_name}/gsc_v2_test_data.npy"
download_file "1NDGE1u5VB9Bk1RMwepb1SJo9o56s4Tx0" "${dnn_name}/gsc_v2_test_label.npy"
download_file "1Mp4RDAzsyNrk4JlUcZBoCBoBiLtK6h-h" "${dnn_name}/gsc_v2_train_data.npy"
download_file "1vn9H8uFtmQWhPm4p0KEcLVrMSKp2ML0V" "${dnn_name}/gsc_v2_train_label.npy"
download_file "1MTrdhQaZ2ZvZvLIOjirE9zgst9wlYdsZ" "${dnn_name}/gsc_v2_validation_data.npy"
download_file "1lHDzS8dX_pqr6ne1tBY9jFfIrXCiZ3jV" "${dnn_name}/gsc_v2_validation_label.npy"

dnn_name="gtsrb"
echo "[5/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1-v81_TTw9yDrCK6KTpsTFKmHNnOBTjUW" "${dnn_name}/GTSRB_test_data.npy"
download_file "15uKYxNbD28fp89y2YXRlMFeElDTo_3_K" "${dnn_name}/GTSRB_test_label.npy"
download_file "1BNIEvoVnRayXXoiTqWirptLqMr2bp7lB" "${dnn_name}/GTSRB_train_data.npy"
download_file "1hCd1NKKDgqsAn-6NE62mGX6jJSegItDj" "${dnn_name}/GTSRB_train_label.npy"

dnn_name="hhar"
echo "[6/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1u5VK6Nf-B9tZTcMtJ3eMpjf5VpLwQY7e" "${dnn_name}/hhar_test_data.npy"
download_file "17TL4Kmf3lozB7pniBDaE4Melz1VkTxo2" "${dnn_name}/hhar_test_label.npy"
download_file "1xeu2y6mHIK0YxVVy-epSR9OIHYwHmiGO" "${dnn_name}/hhar_train_data.npy"
download_file "1Q_6UpDrE_BHZLXlziykQtcxqStw1WeJE" "${dnn_name}/hhar_train_label.npy"

dnn_name="obs"
echo "[7/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1AiGWnABJbJR0DlRUtkUx3BSXxZjyWgRl" "${dnn_name}/obstacle_test_data.npy"
download_file "1h5MG7aaLaw7b83UcUlUwPjRqhxR5dOL7" "${dnn_name}/obstacle_test_label.npy"
download_file "14yqTBhjmCDqhO_Y7gjRmECuBMo9d7GFY" "${dnn_name}/obstacle_train_data.npy"
download_file "1QGmc20vyEYF-5GE7yIyt3mmOJ18WhQd-" "${dnn_name}/obstacle_train_label.npy"
download_file "1y8bHJ9CHBjjZPn8QeK8wBxNCxePMvHDM" "${dnn_name}/obstacle_validation_data.npy"
download_file "1tGc2_h5ppRX3BdFmepFIB1CBRvQ7qsLH" "${dnn_name}/obstacle_validation_label.npy"

dnn_name="svhn"
echo "[8/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1hHzfZ8hVPE8EAyNG5cdD4BMyO3cyVp1s" "${dnn_name}/svhn_test_data.npy"
download_file "1NYuPreAb-1wxzYwi5WcnWq4S8rEXe8Sk" "${dnn_name}/svhn_test_label.npy"
download_file "1BxFLHEy8_hCmAK9QXcvcHLcpDVR2P0t5" "${dnn_name}/svhn_train_data.npy"
download_file "1oFm06MFmjYFAn1NkjxDTH_aW5D2_TCE5" "${dnn_name}/svhn_train_label.npy"
download_file "1c1-47lIsycqgGWRuBKs3vTC6DLxMyrom" "${dnn_name}/svhn_validation_data.npy"
download_file "1SB06lrXDB0rF1hK5CpdSn9E1XgIYCdjV" "${dnn_name}/svhn_validation_label.npy"

dnn_name="us8k"
echo "[9/9] Downloading the ${dnn_name} dataset..."
if [ ! -d ${dnn_name} ]; then
	mkdir ${dnn_name}
fi
download_file "1hWtUljFOq2bvTb-BwS-qFBKP_vcKAPJv" "${dnn_name}/US8K_test_data.npy"
download_file "1m_GY_xt0zy3tJM3GgVTmHV1KgUUIQ3U9" "${dnn_name}/US8K_test_label.npy"
download_file "16n1dVCtFOYckfp800o4i080F5h787CzY" "${dnn_name}/US8K_train_data.npy"
download_file "1p5FCC3uhdskNsTqrbO0FzWQMjJICpMor" "${dnn_name}/US8K_train_label.npy"

rm cookie
