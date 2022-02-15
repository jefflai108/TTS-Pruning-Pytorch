

rate=$1

#./run_transformer-omp.sh --tts_prune_rate $rate --tag pretrained-valid.loss.best-OMP_at_${rate} --stage 6 --stop-stage 6 --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml

for inf_model in valid.loss.best valid.loss.ave; do 
    ./run_transformer-omp.sh --tts_prune_rate $rate --tag pretrained-valid.loss.best-OMP_at_${rate} --stage 7 --stop-stage 7 --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml --inference_model ${inf_model}.pth
    sleep 5s
    ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-OMP_at_${rate}/decode_${inf_model}/
done 


