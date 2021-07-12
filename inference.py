import os, glob, librosa
import tensorflow as tf
import numpy as np
from util.hparams import *
from models.gan import Generator
import soundfile as sf


def test_step(wav_path, idx):
    wav, sr = librosa.core.load(wav_path, sr=sample_rate)
    mel_basis = librosa.filters.mel(sr, n_fft=n_fft, n_mels=mel_dim)
    spectrogram = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel = np.dot(mel_basis, np.abs(spectrogram)).astype(np.float32).T
    mel = np.expand_dims(mel, axis=0)
    audio = g(mel)
    audio = np.squeeze(audio) * 32768

    sf.write(os.path.join(save_dir, 'generated-{}.wav'.format(idx)),
             audio.astype('int16'),
             sample_rate)


save_dir = 'output'
os.makedirs(save_dir, exist_ok=True)

g = Generator()
checkpoint = tf.train.Checkpoint(g=g)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

mel_list = glob.glob(os.path.join('./test', '*.wav'))

for i, wav in enumerate(mel_list):
    test_step(wav, i)