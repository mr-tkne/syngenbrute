import sys
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
import numpy as np
import librosa


FRAMESIZE = 512
FRAMESTEP = 128
HAMLEN = 128
NFILTERS = 24

NFFT = 512
NCEPS = NFILTERS

SIGNIFICANT_MELS = 24


def hz2mel(hz):
    return 2595 * math.log10(1 + hz/700)


def mel2hz(mel):
    return 700 * (10**(mel/2595) - 1)


def tri_filter(data_in, freq, spread):
    data = [x for x in data_in]
    tmp = max(freq-spread, 0)
    for i in range(tmp):
        data[i] = 0
    for i in range(len(data) - freq - spread):
        data[len(data)-i-1] = 0
    for i in range(spread):
        mul = 1 - i/spread
        data[freq-i-1] *= mul
        data[freq+i] *= mul
    return data


def trimpad(sample, startThreshold):
    start = 0
    buffer = np.array(sample)
    normalized = (buffer - buffer.min()) / np.ptp(buffer) - 0.5
    for (idx, smp) in enumerate(normalized):
        # print(smp)
        if abs(smp) > startThreshold:
            start = idx
            break
    return normalized[start:]


def chi2_distance(A, B):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10)
                       for (a, b) in zip(A, B)])
    if math.isnan(chi):
        return 0.
    return chi


class MFCCComparator:
    def __init__(self, target_wav):
        self.sample_rate = 48000
        (length, mfcc) = self.wav2mfcc(target_wav)
        self.target_length = length
        # self.target_mfcc = self.normalize(mfcc)
        self.target_mfcc = librosa.power_to_db(mfcc, ref=np.max) #np.transpose(self.normalize(mfcc))

    def normalize(self, mfcc):
        xmin = 1e9999
        xmax = -1e9999
        for row in mfcc:
            rmin = min(row)
            rmax = max(row)
            if rmin < xmin:
                xmin = rmin
            if rmax > xmax:
                xmax = rmax
        d = xmax - xmin
        for row in mfcc:
            row = (row - xmin)/d
        return mfcc

    def wav2mfcc(self, wav):
        sample_rate, signal = wavfile.read(wav)
        normalized = trimpad(signal, 0.01)
        return self.wavbuf2mfcc(normalized, sample_rate)

    def wavbuf2mfcc(self, buffer, sample_rate):
        # if stereo, get the left channel
        # print(buffer)
        if len(buffer.shape) > 1:
            buffer = np.array([s[0] for s in buffer])
        buffer = (buffer - buffer.min()) / np.ptp(buffer) - 0.5
        # print(buffer)
        mfcc = librosa.feature.mfcc(y=buffer, sr=sample_rate, n_mfcc=20)
        return (buffer.shape[0], mfcc)

    def wavbuf2mfcc_old(self, buffer, sample_rate):
        signal = buffer
        signal_len = len(signal)
        n_frames = int(np.ceil(float(np.abs(signal_len - FRAMESIZE)) / FRAMESTEP))
        pad_signal_len = n_frames * FRAMESTEP + FRAMESIZE
        z = np.zeros(pad_signal_len - signal_len)
        pad_signal = np.append(signal, z)

        indices = (np.tile(np.arange(0, FRAMESIZE), (n_frames, 1))
                   + np.tile(np.arange(0, n_frames * FRAMESTEP, FRAMESTEP),
                             (FRAMESIZE, 1)).T)
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(FRAMESIZE)

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1./NFFT) * (mag_frames ** 2))

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, NFILTERS + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((NFILTERS, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, NFILTERS + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(NCEPS + 1)]
        return (signal_len, mfcc)

    def get_first_mfcc_peak(self, mfcc_slice):
        mel1 = [x[1] for x in mfcc_slice]
        avg_volume = np.average(mel1)
        start = 0
        for (i, v) in enumerate(mel1):
            if v > avg_volume:
                start = i
                break
        return start

    def get_sliding_fn(self, slice, pos):
        k = (1, .6, .1)
        res = [0 for i in range(len(slice[pos]))]
        if pos > 0:
            for i in range(len(res)):
                res[i] += k[1] * slice[pos-1][i]
        if pos > 1:
            for i in range(len(res)):
                res[i] += k[2] * slice[pos-2][i]
        for i in range(len(res)):
            res[i] += k[0] * slice[pos][i]
        if len(slice) > pos+1:
            for i in range(len(res)):
                res[i] += k[1] * slice[pos+1][i]
        if len(slice) > pos+2:
            for i in range(len(res)):
                res[i] += k[2] * slice[pos+2][i]
        return res

    def compare_mfcc(self, reference, compare_to):
        reference = np.transpose(self.normalize(reference))
        compare_to = np.transpose(self.normalize(compare_to))
        min_len = min(len(reference), len(compare_to))
        set1 = reference[:min_len]
        set2 = compare_to[:min_len]
        return abs(chi2_distance(set2, set1))

    def compare_to(self, wav, samplerate=48000):
        # Match by first peak
        # Normalize values? TODO
        # compare using .1 .6 1 .6 .1 sliding windows
        (test_len, test_mfcc) = self.wavbuf2mfcc(wav, samplerate)
        # (target_len, target_mfcc) = self.wav2mfcc('saw.wav')
        # (test_len, test_mfcc) = self.wav2mfcc(wav)
        # mfcc0 = librosa.power_to_db(test_mfcc, ref=np.max)
        mfcc1 = librosa.power_to_db(test_mfcc, ref=np.max)
        return self.compare_mfcc(self.target_mfcc, mfcc1)


if __name__ == '__main__':
    comp = MFCCComparator('saw.wav')

    # sample_rate, signal = wavfile.read('specimenx/feedback40.wav')
    # diff_self = comp.compare_to(signal, sample_rate)
    diff_self = comp.compare_to('specimenx/feedback40.wav')

    # sample_rate, signal = wavfile.read('tinysaw.wav')
    # diff_other1 = comp.compare_to(signal, sample_rate)

    # sample_rate, signal = wavfile.read('120.wav')
    # diff_other2 = comp.compare_to(signal, sample_rate)

    print(diff_self)

    # comp = MFCCComparator('tinysaw.wav')
    # sample_rate, signal = wavfile.read('saw.wav')
    # diff = comp.compare_to(signal, sample_rate)
    # sample_rate, signal = wavfile.read('sine20hz.wav')
    # diff2 = comp.compare_to(signal, sample_rate)
    # print(diff, diff2)
