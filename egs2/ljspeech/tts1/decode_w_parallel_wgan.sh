
stage=$1
text2mel=$2
vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
outname=wav_parallel_wavegan.v3

if [ $stage -eq -1920 ]; then 
    # synthesize extra data 
    for set in train_clean_100; do 
    parallel-wavegan-decode \
        --checkpoint $vocoder_ckpt \
        --feats-scp ${text2mel}/${set}/norm/feats.scp \
        --outdir ${text2mel}/${set}/${outname}/
    done 
    exit 0
fi 


if [ $stage -ge 0 ]; then 
    # text2mel: pretrained tacotron2 provided by espnet 
    # mel2wav: pretrained parallel_wavegan provided by espnet

    text2mel=exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_pretrained_ckpt
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl
    outname=wav_parallel_wavegan.v1

    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3

    vocoder_ckpt=exp/pruned_vocoder/train_nodev_ljspeech_parallel_wavegan.v1-from_pretrained-prune_at_0.7/checkpoint-400000steps.pkl
    outname=wav_parallel_wavegan.v1_prune_at_0.7
fi 

if [ $stage -eq 1 ]; then 
    # text2mel: pretrained transformer-TTS provided by espnet 
    # mel2wav: pretrained parallel_wavegan provided by espnet

    text2mel=exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_pretrained_ckpt
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl
    outname=wav_parallel_wavegan.v1

    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3
fi 

if [ $stage -eq 2 ]; then 
    # text2mel: tacotron2 trained by me (decode_valid.loss.ave, decode_train.loss.ave)
    # mel2wav: pretrained parallel_wavegan provided by espnet
    text2mel=exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.ave
    text2mel=exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.ave
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3
fi 

if [ $stage -eq 3 ]; then 
    # text2mel: transformer-TTS trained by me (decode_valid.loss.ave)
    # mel2wav: pretrained parallel_wavegan provided by espnet
    text2mel=exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.ave
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3
fi 

if [ $stage -eq 4 ]; then 
    # text2mel: tacotron2 trained by me (decode_valid.loss.best, decode_train.loss.best)
    # mel2wav: pretrained parallel_wavegan provided by espnet
    text2mel=exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best
    text2mel=exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.best
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3
fi 

if [ $stage -eq 5 ]; then 
    # text2mel: transformer-TTS trained by me (decode_valid.loss.best)
    # mel2wav: pretrained parallel_wavegan provided by espnet
    text2mel=exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best
    vocoder_ckpt=pretrained/parallel_wavegan/ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl
    outname=wav_parallel_wavegan.v3
fi 

for set in dev eval1; do 
    parallel-wavegan-decode \
        --checkpoint $vocoder_ckpt \
        --feats-scp ${text2mel}/${set}/norm/feats.scp \
        --outdir ${text2mel}/${set}/${outname}/
done 

