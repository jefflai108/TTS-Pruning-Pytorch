import librosa

def calculate_duration(file):
    with open(file, 'r') as f:
        content = f.readlines()
    content = [x.strip('\n').split()[1] for x in content]

    duration = 0 
    for wav in content:
        duration += librosa.get_duration(filename=wav)
    print('duration for %s is %.2f seconds' % (file, duration))

if __name__ == '__main__':
    calculate_duration('data/tr_no_dev-10%/wav.scp')
    calculate_duration('data/tr_no_dev-20%/wav.scp')
    calculate_duration('data/tr_no_dev-30%/wav.scp')
    calculate_duration('data/tr_no_dev-40%/wav.scp')
    calculate_duration('data/tr_no_dev-50%/wav.scp')
    calculate_duration('data/tr_no_dev-60%/wav.scp')
    calculate_duration('data/tr_no_dev-70%/wav.scp')
    calculate_duration('data/tr_no_dev-80%/wav.scp')
    calculate_duration('data/tr_no_dev-90%/wav.scp')
    calculate_duration('data/tr_no_dev/wav.scp')

