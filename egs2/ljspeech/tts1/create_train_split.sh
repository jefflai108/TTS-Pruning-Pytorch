
mkdir -p data/tr_no_dev-10%
for item in text utt2spk wav.scp; do
    head -n 1260 data/tr_no_dev/$item > data/tr_no_dev-10%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-10%

mkdir data/tr_no_dev-20%
for item in text utt2spk wav.scp; do
    head -n 2520 data/tr_no_dev/$item > data/tr_no_dev-20%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-20%

mkdir -p data/tr_no_dev-30%
for item in text utt2spk wav.scp; do
    head -n 3780 data/tr_no_dev/$item > data/tr_no_dev-30%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-30%

mkdir -p data/tr_no_dev-40%
for item in text utt2spk wav.scp; do
    head -n 5040 data/tr_no_dev/$item > data/tr_no_dev-40%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-40%

mkdir -p data/tr_no_dev-50%
for item in text utt2spk wav.scp; do
    head -n 6300 data/tr_no_dev/$item > data/tr_no_dev-50%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-50%

mkdir -p data/tr_no_dev-60%
for item in text utt2spk wav.scp; do
    head -n 7560 data/tr_no_dev/$item > data/tr_no_dev-60%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-60%

mkdir -p data/tr_no_dev-70%
for item in text utt2spk wav.scp; do
    head -n 8820 data/tr_no_dev/$item > data/tr_no_dev-70%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-70%

mkdir -p data/tr_no_dev-80%
for item in text utt2spk wav.scp; do
    head -n 10080 data/tr_no_dev/$item > data/tr_no_dev-80%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-80%

mkdir -p data/tr_no_dev-90%
for item in text utt2spk wav.scp; do
    head -n 11340 data/tr_no_dev/$item > data/tr_no_dev-90%/$item 
done 
utils/fix_data_dir.sh data/tr_no_dev-90%
