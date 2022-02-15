import os, sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

EXPDIR = '/data/sls/temp/clai24/lottery-ticket/espnet/egs2/ljspeech/tts1'

def extract_f0_from_audio(filename):
    y, sr = librosa.load(filename)
    assert sr == 22050
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    utt_duration = len(y)/sr
    f0_mean = np.mean(f0[~np.isnan(f0)])
    f0_std  = np.std(f0[~np.isnan(f0)])
 
    return utt_duration, (f0_mean, f0_std)


def run_utterance(utt, sparsity, utt_cnt, diff_tuple):
        
    gt_file = '/data/sls/temp/clai24/data/LJSpeech-1.1/wavs/' + utt + '.wav'
    gt_utt_duration, (gt_f0_mean, gt_f0_std) = extract_f0_from_audio(gt_file)
    
    #unpruned_file = os.path.join(EXPDIR, 'exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav') 
    unpruned_file = os.path.join(EXPDIR, 'exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav')
    #pruned_file = os.path.join(EXPDIR, 'exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/', 'tts_pretrained-valid.loss.best-prune_at_' + str(sparsity), 'decode_valid.loss.best/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav')     
    #pruned_file = os.path.join(EXPDIR, 'exp/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space/', 'tts_pretrained-valid.loss.best-OMP_at_' + str(sparsity), 'decode_valid.loss.best/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav')     
    #pruned_file = os.path.join(EXPDIR, 'exp/prune_tts2/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/', 'tts_pretrained-valid.loss.best-prune_at_0.8-0.9/decode_latest/dev/wav_parallel_wavegan.v1_prune_at_' + str(sparsity), utt + '_gen.wav')
    if not os.path.exists(unpruned_file):
        return diff_tuple, utt_cnt
    utt_cnt += 1
    unpruned_utt_duration, (unpruned_f0_mean, unpruned_f0_std) = extract_f0_from_audio(unpruned_file)
    utt_duration_diff = unpruned_utt_duration - gt_utt_duration
    f0_mean_diff = unpruned_f0_mean - gt_f0_mean
    diff_tuple = (diff_tuple[0] + utt_duration_diff, diff_tuple[1] + f0_mean_diff)

    #pruned_utt_duration, (pruned_f0_mean, pruned_f0_std) = extract_f0_from_audio(pruned_file)
    #utt_duration_diff = pruned_utt_duration - gt_utt_duration
    #f0_mean_diff = pruned_f0_mean - gt_f0_mean
    #diff_tuple = (diff_tuple[0] + utt_duration_diff, diff_tuple[1] + f0_mean_diff)
    
    return diff_tuple, utt_cnt 
	
if __name__ == '__main__':

    utt_cnt = 0
    diff_tuple = (0, 0) # (utt_duration_diff, f0_mean_diff)
    sparsity = sys.argv[1]
    for split in ['dev', 'eval1']:
        for filename in os.listdir(os.path.join(EXPDIR, 'exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/decode_valid.loss.best/', split, 'wav_parallel_wavegan.v3/')):
            utt = filename.split('_')[0]
            print(utt)
            diff_tuple, utt_cnt = run_utterance(utt, sparsity, utt_cnt, diff_tuple)
            print(diff_tuple)
            
    print('results are:')
    print(diff_tuple[0]/utt_cnt, diff_tuple[1]/utt_cnt)
