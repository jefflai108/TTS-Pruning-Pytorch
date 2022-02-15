

stage=$1


if [ $stage -eq -100 ]; then
    ./run_transformer.sh \
        --train_set tr_no_dev_and_train_clean_100-syn \
        --stage 2 --stop-stage 5 \
        --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev_and_train_clean_100-syn \
        --dumpdir dump2

    ./run_transformer.sh \
        --train_set train_clean_100-syn \
        --stage 2 --stop-stage 5 \
        --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-train_clean_100-syn \
        --dumpdir dump2

    ./run_transformer-prune.sh --train_set tr_no_dev_and_train_clean_100-syn --tts_prune_rate 0.9 --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev_and_train_clean_100-syn --stage 6 --stop-stage 6 --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml --dumpdir dump2


    #./run_transformer.sh \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 1 --train_config conf/tuning/train_transformer.yaml
fi 

if [ $stage -eq -900 ]; then 
    for init_rate in 0.85; do
    for target_rate in 0.95 0.9; do 
        ./run_transformer-prune_progressive.sh \
            --tts_prune_rate ${target_rate} \
            --tag pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate} \
            --stage 7 --stop-stage 7 \
            --inference_model latest.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate}/decode_latest/
    done 
    done 
fi 

if [ $stage -eq -901 ]; then 
    for init_rate in 0.8; do
    for target_rate in 0.95 0.9; do 
        ./run_transformer-prune_progressive.sh \
            --tts_prune_rate ${target_rate} \
            --tag pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate} \
            --stage 7 --stop-stage 7 \
            --inference_model latest.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate}/decode_latest/
    done 
    done 

    for init_rate in 0.9; do
    for target_rate in 0.95; do 
        ./run_transformer-prune_progressive.sh \
            --tts_prune_rate ${target_rate} \
            --tag pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate} \
            --stage 7 --stop-stage 7 \
            --inference_model latest.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${init_rate}-${target_rate}/decode_latest/
    done 
    done 

fi 

if [ $stage -eq -579 ]; then 
    for rate in 0.3; do
        for inf_model in valid.loss.best valid.loss.ave; do 
            ./run_transformer-prune.sh \
                --tts_prune_rate $rate \
                --tag pretrained-valid.loss.best-prune_at_${rate} \
                --stage 7 --stop-stage 7 \
                --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
                --inference_model ${inf_model}.pth
            sleep 5s
            ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}/decode_${inf_model}/
        done 
    done 
fi 

if [ $stage -eq -578 ]; then 
    #for rate in 0.3 0.2 0.1 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95; do
    for rate in 0.3 0.2 0.1 0.4; do
        for inf_model in valid.loss.best valid.loss.ave; do 
            ./run_transformer-prune.sh \
                --train_set tr_no_dev_and_train_clean_100-syn \
                --tts_prune_rate $rate \
                --tag pretrained-valid.loss.best-prune_at_${rate}-tr_no_dev_and_train_clean_100-syn \
                --stage 7 --stop-stage 7 \
                --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
                --dumpdir dump2 \
                --inference_model ${inf_model}.pth
            sleep 5s
            ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}-tr_no_dev_and_train_clean_100-syn/decode_${inf_model}/
        done 
    done 
fi 

if [ $stage -eq -789 ]; then 
    for rate in 0.7 0.9; do
        for inf_model in valid.loss.best valid.loss.ave; do 
            ./run_transformer_teacher-prune.sh \
                --tts_prune_rate $rate \
                --tag pretrained-valid.loss.best-prune_at_${rate}-with_teacher \
                --stage 7 --stop-stage 7 \
                --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
                --inference_model ${inf_model}.pth
            sleep 5s
            ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_${rate}-with_teacher/decode_${inf_model}/
        done 
    done 
fi 

if [ $stage -eq -1 ]; then 
    # prepare Librispeech synthesis data for self-training 
    #./run_transformer.sh \
    #    --test_sets train_clean_100 \
    #    --stage 2 --stop-stage 3 \
    #    --ngpu 1 --train_config conf/tuning/train_transformer.yaml

    # forward pass to get synthesis speech
    ./run_transformer.sh \
        --test_sets train_clean_100 \
        --stage 7 --stop-stage 7 \
        --ngpu 1 --train_config conf/tuning/train_transformer.yaml 
        --inference_model valid.loss.best.pth
fi 

if [ $stage -ge 10 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-10% \
    #    --tag pretrained-valid.loss.best-prune_at_0.7-tr_no_dev-10% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-10% 
    #
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-10% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-10% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-10%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-10% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-10% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-10% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -100 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-10%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 20 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-20% \
    #    --tag pretrained-valid.loss.best-prune_at_0.7-tr_no_dev-20% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-20% 
    #
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-20% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-20% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-20%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-20% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-20% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-20% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -200 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-20%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 30 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-30% \
    #    --tag pretrained-valid.loss.best-prune_at_0.7-tr_no_dev-30% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-30% 
    #sleep 5s
    #
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-30% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-30% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-30%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-30% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-30% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-30% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -300 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-30%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 40 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-40% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-40% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-40% 
    #
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-40% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-40% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-40%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-40% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-40% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-40% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -400 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-40%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 50 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-50% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-50% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-50% 
    
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-50% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-50% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-50%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-50% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-50% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-50% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -500 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-50%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 60 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-60% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-60% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-60% 
    
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-60% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-60% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-60%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-60% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-60% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-60% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -600 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-60%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 70 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-70% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-70% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-70% 
    
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-70% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-70% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-70%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-70% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-70% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-70% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -700 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-70%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 80 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-80% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-80% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-80% 
    #sleep 5s
    #
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-80% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-80% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-80%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-80% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-80% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-80% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -800 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-80%/decode_${inf_model}/
    done 
fi 

if [ $stage -ge 90 ]; then 
    #./run_transformer.sh \
    #    --train_set tr_no_dev-90% \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-90% \
    #    --stage 2 --stop-stage 5 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-90% 
    #sleep 5s
   
    #./run_transformer-prune.sh \
    #    --train_set tr_no_dev-90% \
    #    --tts_prune_rate 0.9 \
    #    --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-90% \
    #    --stage 6 --stop-stage 6 \
    #    --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
    #    --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-90%

    for inf_model in valid.loss.best valid.loss.ave; do 
        ./run_transformer-prune.sh \
            --train_set tr_no_dev-90% \
            --tts_prune_rate 0.9 \
            --tag pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-90% \
            --stage 7 --stop-stage 7 \
            --ngpu 4 --train_config conf/tuning/train_transformer-4gpu.yaml \
            --tts_stats_dir exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_stats_raw_phn_tacotron_g2p_en_no_space-tr_no_dev-90% \
            --inference_model ${inf_model}.pth
        sleep 5s
        ./decode_w_parallel_wgan.sh -900 exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-90%/decode_${inf_model}/
    done 
fi 


