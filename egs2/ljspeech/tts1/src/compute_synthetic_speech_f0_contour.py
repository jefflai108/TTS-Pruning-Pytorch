import os, sys
import numpy as np 
import librosa 
import librosa.display
import matplotlib.pyplot as plt

def extract_f0_from_audio(filename):
    y, sr = librosa.load(filename)
    assert sr == 22050
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    utt_duration = len(y)/sr

    return utt_duration, (y, f0, times)

def plot_f0_contour(fig, ax, utt_name, gt_file, pruned_files, plotting_dir):
    
    # extract spectrogram (from ground truth utterance) as the background img 
    gt_utt_duration, (gt_y, gt_f0, gt_times) = \
        extract_f0_from_audio(gt_file)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(gt_y)), ref=np.max)
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)

    ax.set(title='Utterance ' + utt_name + ' f0 estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(gt_times, gt_f0, label='natural f0', color='cyan', linewidth=3)
   
    colors = ['lawngreen', 'yellowgreen', 'yellow', 'ivory', 'gold', 'wheat', 'orange', 'darkorange', 'orangered']
    for kk, (name, pruned_file) in enumerate(pruned_files):   
        pruned_utt_duration, (pruned_y, pruned_f0, pruned_times) = \
            extract_f0_from_audio(pruned_file)
        ax.plot(pruned_times, pruned_f0, label=name + ' f0', color=colors[kk], linewidth=3)

    ax.legend(loc='upper left', fontsize=10)
    plt.savefig(os.path.join(plotting_dir, 'shit3.png'))


def process_ljspeech_csv(split):

    if split == 'dev':
        text = '/data/sls/temp/clai24/lottery-ticket/espnet/egs2/ljspeech/tts1/dump/raw/dev/text'
    elif split == 'eval1':
        text = '/data/sls/temp/clai24/lottery-ticket/espnet/egs2/ljspeech/tts1/dump/raw/eval1/text'
    with open(text, 'r') as f: 
        df = f.readlines() 
    df = {x.strip('\n').split(' ')[0]:' '.join(x.strip('\n').split(' ')[1:]) for x in df}

    # remove final punctuation 
    df = {k:v[:-1] if v[-1] in string.punctuation else v for k,v in df.items()}
    
    return df

def recognize_an_audio(audio_path, r, utt2ground_truth, ground_truths, hypotheses):

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)  # read the entire audio file

    # get the ground truth 
    if '_gen' in audio_path:
        audio_path = audio_path.replace('_gen', '')
    if audio_path.split('/')[-1].split('.wav')[0] not in utt2ground_truth.keys():
        return ground_truths, hypotheses
    ground_truth = utt2ground_truth[audio_path.split('/')[-1].split('.wav')[0]]


    try:
        hypothesis = r.recognize_google(audio, language='en-US')
    except Exception as e:
        ground_truths.append(text_cleaners.english_cleaners(ground_truth))
        hypotheses.append("")
        return ground_truths, hypotheses
    
    ground_truths.append(text_cleaners.english_cleaners(ground_truth))
    hypotheses.append(text_cleaners.english_cleaners(hypothesis))
    #print('ground_truths is', ground_truths)
    #print('hypotheses is', hypotheses)
    return ground_truths, hypotheses

def run(synthesis_directory, split):

    utt2ground_truth = process_ljspeech_csv(split)
    ground_truths, hypotheses = [], []
    r = sr.Recognizer()
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.Strip(),
    ]) 

    for filename in os.listdir(synthesis_directory):
        if filename.endswith(".wav"):
            audio_file = os.path.join(synthesis_directory, filename)
            #print('processing %s' % audio_file)
            ground_truths, hypotheses = recognize_an_audio(audio_file, r, utt2ground_truth, ground_truths, hypotheses)

    #error = wer(ground_truths, hypotheses, truth_transform=transformation, hypothesis_transform=transformation)
    error = wer(ground_truths, hypotheses)
    print('WER for %s is %f' % (split, error))


if __name__ == '__main__':

    utt = 'LJ050-0009'
    plotting_dir = os.path.join('f0_plots/', utt)
    if not os.path.exists(plotting_dir): os.mkdir(plotting_dir) 

    gt_file = '/data/sls/temp/clai24/data/LJSpeech-1.1/wavs/' + utt + '.wav'
    pruned_files = []
    #for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        name = str(sparsity*100) + '% PARP ' 
        pruned_file = os.path.join('exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/', 'tts_pretrained-valid.loss.best-prune_at_' + str(sparsity), 'decode_valid.loss.best/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav')
        pruned_file = os.path.join('exp/ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space/', 'tts_pretrained-valid.loss.best-OMP_at_' + str(sparsity), 'decode_valid.loss.best/dev/wav_parallel_wavegan.v3/', utt + '_gen.wav')
        pruned_files.append((name, pruned_file))

    fig, ax = plt.subplots()
    plot_f0_contour(fig, ax, utt, gt_file, pruned_files, plotting_dir)
