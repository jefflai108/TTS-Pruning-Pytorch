. ~/.bashrc

#scplocal2satori exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space  /home/clai24/prune_tts/

# transfer pruned transformer-tts to Satori 
#rsync -zarv --exclude="*.pth" --exclude="*.png" --exclude="*.yaml" --exclude="*.wav" --exclude='events.out.*' exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space clai24@satori-login-002.mit.edu:/nobackup/users/clai24/prune_tts/

# transfer pruned Tacotron2 to Satori
#rsync -zarv --exclude="*.pth" --exclude="*.png" --exclude="*.yaml" --exclude="*.wav" --exclude='events.out.*' exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space clai24@satori-login-002.mit.edu:/nobackup/users/clai24/prune_tts/

# transfer pruned vocoder syntheses to SLS
rsync -zarv clai24@satori-login-002.mit.edu:/nobackup/users/clai24/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space exp/prune_tts2/
rsync -zarv clai24@satori-login-002.mit.edu:/nobackup/users/clai24/prune_tts2/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space exp/prune_tts2/
