import sys
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
import numpy as np


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

class MFCCComparator:
    def __init__(self, target_wav):
        self.sample_rate = 48000
        (length, mfcc) = self.wav2mfcc(target_wav)
        self.target_length = length
        self.target_mfcc = self.normalize(mfcc)

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
        return self.wavbuff2mfcc(signal, sample_rate)

    def wavbuff2mfcc(self, buffer, sample_rate):
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

    def compare_to(self, wav):
        # Match by first peak
        # Normalize values? TODO
        # compare using .1 .6 1 .6 .1 sliding windows
        target_mfcc = self.target_mfcc
        (test_len, test_mfcc) = self.wavbuff2mfcc(wav, 48000)

        target_start = self.get_first_mfcc_peak(target_mfcc[0:50])
        test_start = 0 #self.get_first_mfcc_peak(target_mfcc[0:50])

        target_mfcc = target_mfcc[target_start:]
        test_mfcc = self.normalize(test_mfcc[test_start:])

        min_len = min(len(target_mfcc), len(test_mfcc))
        delta_heat = []
        for i in range(min_len):
            target_win = self.get_sliding_fn(target_mfcc, i)
            test_win = self.get_sliding_fn(test_mfcc, i)
            delta_heat.append([a - b for (a, b) in zip(target_win, test_win)])
        return 1/(abs(np.average([np.average(col) for col in delta_heat]))+1)

if __name__ == '__main__':
    comp = MFCCComparator('in.wav')
    diff_self = comp.compare_to('in.wav')
    diff_other1 = comp.compare_to('sine20hz.wav')
    diff_other2 = comp.compare_to('120.wav')
    print(diff_self, diff_other1, diff_other2)
