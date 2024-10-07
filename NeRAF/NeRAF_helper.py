import os 
import torch
import numpy as np
import pyroomacoustics
from scipy.signal import hilbert
import torchaudio
from torch.nn import functional as F
import torch.nn as nn

""" Helper methods for evaluation """

class SpectralLoss(nn.Module):
    """
    Compute a loss between two log power-spectrograms. 
    From  https://github.com/facebookresearch/SING/blob/main/sing/dsp.py#L79 modified

    Arguments:
        base_loss (function): loss used to compare the log power-spectrograms.
            For instance :func:`F.mse_loss`
        epsilon (float): offset for the log, i.e. `log(epsilon + ...)`
        **kwargs (dict): see :class:`STFT`
    """

    def __init__(self, base_loss=F.mse_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='mag', **kwargs):
        super(SpectralLoss, self).__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.dB = dB
        self.stft_input_type = stft_input_type
        self.reduction = reduction

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def _log_spectrogram(self, STFT):
        if self.dB and self.stft_input_type == 'mag':
            return 10*torch.log10(self.epsilon + STFT) 
        elif not self.dB and self.stft_input_type == 'mag':
            return torch.log(self.epsilon + STFT)
        elif self.stft_input_type == 'log mag': 
            return STFT

    def forward(self, a, b):
        spec_a = self._log_spectrogram(a)
        spec_b = self._log_spectrogram(b)
        return self.base_loss(spec_a, spec_b, reduction=self.reduction)

def compute_t60(true_in, gen_in, fs, advanced = False):
    ch = true_in.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        try:
            if advanced: 
                true = measure_rt60_advance(true_in[c], sr=fs)
                gen = measure_rt60_advance(gen_in[c], sr=fs)
            else:
                true = pyroomacoustics.experimental.measure_rt60(true_in[c], fs=fs, decay_db=30)
                gen = pyroomacoustics.experimental.measure_rt60(gen_in[c], fs=fs, decay_db=30)
        except:
            true = -1
            gen = -1
        gt.append(true)
        pred.append(gen)
    return np.array(gt), np.array(pred)

def measure_rt60_advance(signal, sr, decay_db=10, cutoff_freq=200):
    # following RAF implementation
    signal = torch.from_numpy(signal)
    signal = torchaudio.functional.highpass_biquad(
        waveform=signal,
        sample_rate=sr,
        cutoff_freq=cutoff_freq
    )
    signal = signal.cpu().numpy()
    rt60 = pyroomacoustics.experimental.measure_rt60(signal, sr, decay_db=decay_db, plot=False)
    return rt60

def Envelope_distance(predicted, gt):
    ch = predicted.shape[0]
    envelope_distance=0
    for c in range(ch):
        pred_env = np.abs(hilbert(predicted[c,:]))
        gt_env = np.abs(hilbert(gt[c,:]))
        distance = np.sqrt(np.mean((gt_env - pred_env)**2))
        envelope_distance += distance
    return float(envelope_distance)

def SNR(predicted, gt):
    mse_distance = np.mean(np.power((predicted - gt), 2))
    snr = 10. * np.log10((np.mean(gt**2) + 1e-4) / (mse_distance + 1e-4))
    return float(snr)

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def Magnitude_distance(predicted_mag, gt_mag):
    ch = predicted_mag.shape[0]
    stft_mse = 0
    for c in range(ch): 
        stft_mse += np.mean(np.power(predicted_mag[c] - gt_mag[c], 2))
    return float(stft_mse)
    
def measure_clarity(signal, time=50, fs=44100):
    h2 = signal**2
    t = int((time/1000)*fs + 1) 
    return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))

def evaluate_clarity(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_clarity = measure_clarity(np_pred_ir[c,...], fs=fs)
        gt_clarity = measure_clarity(np_gt_ir[c,...], fs=fs)
        gt.append(gt_clarity)
        pred.append(pred_clarity)
    return np.array(gt), np.array(pred)

def measure_edt(h, fs=44100, decay_db=10):
    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    if np.all(energy == 0):
        return np.nan

    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    # compute the decay time
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time 
    return est_edt

def evaluate_edt(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_edt = measure_edt(np_pred_ir[c], fs=fs)
        gt_edt = measure_edt(np_gt_ir[c], fs=fs)
        gt.append(gt_edt)
        pred.append(pred_edt)
    return np.array(gt), np.array(pred)

