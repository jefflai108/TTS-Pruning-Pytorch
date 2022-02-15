

stage=$1
rate=$2

if [ $stage -eq -578 ]; then 
    for rate in 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 0.99; do
        for inf_model in valid.loss.best valid.loss.ave; do 
            ./run_tacotron2-prune.sh \
                --tts_prune_rate $rate \
                --tag pretrained-valid.loss.best-prune_at_${rate} \
                --stage 7 --stop-stage 7 \
                --inference_model ${inf_model}.pth
            sleep 5s
            ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}/decode_${inf_model}/
        done 
    done 
fi 

./run_tacotron2-prune_progressive.sh --tts_prune_rate 0.85 --tag pretrained-valid.loss.best-prune_at_0.8-0.85 --stage 6 --stop-stage 6 --ngpu 4 --inference_model valid.loss.best.pth

if [ $stage -eq 2 ]; then 
    #./run_tacotron2-omp.sh --tts_prune_rate $rate --tag pretrained-valid.loss.best-OMP_at_${rate} --stage 6 --stop-stage 6 --ngpu 4 
    ./run_tacotron2-omp.sh --tts_prune_rate $rate --tag pretrained-valid.loss.best-OMP_at_${rate} --stage 6 --stop-stage 6 --ngpu 1
    sleep 5s

    #for inf_model in valid.loss.best valid.loss.ave; do 
    #    ./run_tacotron2-omp.sh --tts_prune_rate $rate --tag pretrained-valid.loss.best-OMP_at_${rate} --stage 7 --stop-stage 7 --inference_model ${inf_model}.pth
    #    sleep 5s
    #    ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-OMP_at_${rate}/decode_${inf_model}/
    #done 
fi 
