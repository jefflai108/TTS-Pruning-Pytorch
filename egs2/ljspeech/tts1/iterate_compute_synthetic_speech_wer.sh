#!/bin/bash

for split in dev eval1; do 
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.88 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.85 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.8 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.75 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.7 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.6 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.5 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.4 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.3 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.2 $split
    python src/compute_synthetic_speech_wer.py exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/${split}/wav_parallel_wavegan.v1_prune_at_0.1 $split
done 


#for sparsity in 1 2 3 4 5 6 7 75 8 85 9 95 99; do
#for sparsity in 0.85-0.95 0.8-0.95 0.85-0.9 0.8-0.9 0.9-0.95 0.7-0.95 0.7-0.9 0.7-0.8 0.75-0.8 0.75-0.9 0.75-0.85 0.8-0.85 0.8-0.95; do
#    for split in dev eval1; do 
#        expname=tts_pretrained-valid.loss.best-prune_at_0.${sparsity}
#        expname=tts_pretrained-valid.loss.best-prune_at_${sparsity}
#        echo $expname $split 
#        #python src/compute_synthetic_speech_wer.py exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/${expname}/decode_valid.loss.best/$split/wav_parallel_wavegan.v3/ $split
#        python src/compute_synthetic_speech_wer.py exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/${expname}/decode_latest/$split/wav_parallel_wavegan.v3/ $split
#    done 
#done 

#for sparsity in 1 2 3 4 5 6 7 75 8 85 9 95 99; do
##for sparsity in 0.85-0.95 0.8-0.95 0.85-0.9 0.8-0.9 0.9-0.95 0.7-0.95 0.7-0.9 0.7-0.8; do
##for sparsity in 7 9; do
#    for split in dev eval1; do 
#        #expname=tts_pretrained-valid.loss.best-prune_at_0.${sparsity}
#        #expname=tts_pretrained-valid.loss.best-prune_at_${sparsity}
#        #expname=tts_pretrained-valid.loss.best-OMP_at_0.${sparsity}
#        #expname=tts_random_init-prune_at_0.${sparsity}
#        #expname=tts_pretrained-valid.loss.best-prune_at_0.${sparsity}-with_teacher/
#        #expname=tts_pretrained-valid.loss.best-prune_at_0.${sparsity}-train_clean_100-syn_then_tr_no_dev
#        expname=tts_pretrained-valid.loss.best-prune_at_0.${sparsity}-tr_no_dev_and_train_clean_100-syn
#        echo $expname $split 
#        python src/compute_synthetic_speech_wer.py exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/${expname}/decode_valid.loss.best/$split/wav_parallel_wavegan.v3/ $split
#        #python src/compute_synthetic_speech_wer.py exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/${expname}/decode_latest/$split/wav_parallel_wavegan.v3/ $split
#    done 
#done 

#for percentage in 10 20 30 40 50 60 70 80 90; do
#    for split in dev eval1; do 
#        expname=tts_pretrained-valid.loss.best-prune_at_0.7-tr_no_dev-${percentage}%
#        expname=tts_pretrained-valid.loss.best-prune_at_0.9-tr_no_dev-${percentage}%
#        echo $expname $split 
#        python src/compute_synthetic_speech_wer.py exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/${expname}/decode_valid.loss.best/$split/wav_parallel_wavegan.v3/ $split
#    done 
#done 

