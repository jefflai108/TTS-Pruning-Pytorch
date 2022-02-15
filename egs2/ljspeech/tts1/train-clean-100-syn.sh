
rate=$1
stage=$2

if [ $stage -eq 1 ]; then 
# step 1 training on train_clean_100-syn
./run_transformer-prune.sh \
    --train_set train_clean_100-syn \
    --tts_prune_rate ${rate} \
    --tag pretrained-valid.loss.best-prune_at_${rate}-train_clean_100-syn \
    --stage 6 --stop-stage 6 --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-train_clean_100-syn/ \
    --dumpdir dump2
fi


if [ $stage -eq 2 ]; then 
# step 2 continue training on and only on tr_no_dev
./run_transformer-prune-init-train-clean-100-syn.sh \
    --tts_prune_rate ${rate} \
    --tag pretrained-valid.loss.best-prune_at_${rate}-train_clean_100-syn_then_tr_no_dev \
    --stage 6 --stop-stage 6 --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    --train_args "--init_param exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}-train_clean_100-syn/valid.loss.best.pth"
fi 


if [ $stage -eq 3 ]; then 
for inf_model in valid.loss.best valid.loss.ave; do 
    ./run_transformer-prune-init-train-clean-100-syn.sh \
        --tts_prune_rate $rate \
        --tag pretrained-valid.loss.best-prune_at_${rate}-train_clean_100-syn_then_tr_no_dev \
        --stage 7 --stop-stage 7 \
        --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
        --inference_model ${inf_model}.pth
    sleep 5s
    ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}-train_clean_100-syn_then_tr_no_dev/decode_${inf_model}/
done 
fi 
