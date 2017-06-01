import numpy as np

def select_feature_func(feature_name):
    if feature_name == 'aqibsaeed_1':
        return get_feature_aqibsaeed_1
    elif feature_name == 'mfcc':
        return get_feature_mfcc

def get_feature_mfcc(X, sr, n_mfcc=13):
    import librosa
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def get_feature_aqibsaeed_1(X, sr, au_path=None):
    """
    http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
    """
    import librosa
    if au_path is not None:
        X, sr = librosa.load(au_path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T,axis=0)
    feature = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    return feature

def get_feature_aqibsaeed_conv(X, sr, au_path=None):
    """
    http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
    """
    import librosa
    def windows(data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size / 2)
    bands = 60
    frames = 41
    window_size = 512 * (frames - 1)
    for (start,end) in windows(X, window_size):
        if(len(X[start:end]) == window_size):
            signal = X[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.logamplitude(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
