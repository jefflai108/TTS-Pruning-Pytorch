import speech_recognition as sr
import os, sys
import jiwer
from jiwer import wer
import string 
import text_cleaners 

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

    #run('/data/sls/temp/clai24/data/LJSpeech-1.1/wavs', 'dev')
    run(sys.argv[1], sys.argv[2])
