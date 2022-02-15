#exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best/train_clean_100/wav_parallel_wavegan.v3/103-1240-0006_gen.wav

import os 
syn_root = 'exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best/train_clean_100/wav_parallel_wavegan.v3/'

with open('data/train_clean_100-syn/wav.scp', 'r') as f:
    content = f.readlines()
content = [x.strip('\n').split()[0] for x in content]

with open('data/train_clean_100-syn/wav.scp', 'w') as f:
    for i in content:
        path = os.path.join(syn_root, i + '_gen.wav')
        f.write('%s %s\n' % (i, path))


#with open('data/train_clean_100-syn/utt2spk', 'r') as f:
#    content = f.readlines()
#content = [x.strip('\n') for x in content]
#
#
#with open('data/train_clean_100-syn/utt2spk', 'w') as f:
#    for i in content:
#        f.write('%s %s\n' % (i.split()[0], 'LJ'))
