import torch
from torch import nn
import torch.nn.functional as F
import librosa
import numpy as np
import pyroomacoustics
from scipy.signal import hilbert


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self, loss_type='l1'):
        """Initilize log STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, x_log, y_log):
        """Calculate forward propagation.

        Args:
            x_log (Tensor): Log magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_log (Tensor): Log magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        target = y_log
        pred = x_log
        if self.loss_type == 'l1':
            return F.l1_loss(target, pred)
        elif self.loss_type == 'mse':
            return F.mse_loss(target, pred)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, loss_type = 'l1'
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(loss_type=loss_type)

    def forward(self, x_log, y_log):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        # from log to mag STFT
        x_mag = torch.exp(x_log) - 1e-3
        y_mag = torch.exp(y_log) - 1e-3
        # apply losses
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_log, y_log)

        return 1e-1*sc_loss+mag_loss


class NeRAFEvaluator(object):
    def __init__(self, fs=44100):
        self.fs = fs # sample rate
        #self.spectral_loss_NAF = SpectralLoss(base_loss=F.l1_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='log mag')
    
    def get_full_metrics(self, mag_prd, mag_gt, wav_gt_ff, wav_pred_istft, wav_gt_istft,  log_prd, log_gt):#, gl=False):
        
        # Compute Spectral metric used in NAF
        #NAF_spectral_loss = self.spectral_loss_NAF(torch.from_numpy(log_prd), torch.from_numpy(log_gt))

        # Get waveform from mag STFT
        wav_prd = wav_pred_istft
        wav_gt = wav_gt_istft
        n_ch = wav_gt.shape[0]

        # zero pad waveform to be the same size as gt ff (i.e. max_len * hop_len )
        wav_prd = np.pad(wav_prd, ((0,0),(0, wav_gt_ff.shape[1]-wav_prd.shape[1])), 'constant', constant_values=(0,0))
        wav_gt = np.pad(wav_gt, ((0,0),(0, wav_gt_ff.shape[1]-wav_gt.shape[1])), 'constant', constant_values=(0,0))

        ## Waveform related
        env_loss = Envelope_distance(wav_prd, wav_gt_ff)

        # Compute t60 error, edt and c50 on gt from file
        t60s_gt, t60s_prd = compute_t60(wav_gt_ff, wav_prd, fs=self.fs)
        t60s = np.concatenate((t60s_gt, t60s_prd))
        t60s = np.expand_dims(t60s, axis=0)
        diff = np.abs(t60s[:,n_ch:]-t60s[:,:n_ch])/np.abs(t60s[:,:n_ch])
        mask = np.any(t60s<-0.5, axis=1)
        diff = np.mean(diff, axis=1)
        diff[mask] = 1
        mean_t60error = np.mean(diff)*100
        invalid = np.sum(mask)
        
        edt_gt, edt_prd = evaluate_edt(wav_prd, wav_gt_ff, fs=self.fs)
        edts = np.concatenate((edt_gt, edt_prd))
        edt_instance = np.abs(edts[n_ch:]-edts[:n_ch]) # pred-gt
        mean_edt = np.mean(edt_instance, axis=0) # mean over instance channels

        c50_gt, c50_prd = evaluate_clarity(wav_prd, wav_gt_ff, fs=self.fs)
        c50s = np.concatenate((c50_gt, c50_prd))
        c50_instance = np.abs(c50s[n_ch:]-c50s[:n_ch]) # pred-gt
        mean_c50 = np.mean(c50_instance, axis=0) # mean over instance channels

        res = {
                "audio_env": env_loss,
                "audio_T60_mean_error": mean_t60error,
                "audio_total_invalids_T60": invalid,
                "audio_EDT": mean_edt,
                "audio_C50": mean_c50,
                }
        
        #transform to float
        for key in res.keys():
            #if tensor go to numpy
            if isinstance(res[key], torch.Tensor):
                res[key] = res[key].item()
            else:
                res[key] = float(res[key])

        return res
        
    
    def get_stft_metrics(self,mag_prd,mag_gt):
        ## STFT related
        mag_loss = np.mean(np.power(mag_prd - mag_gt, 2)) * 2

        return {
                "audio_mag": mag_loss,
        }

        



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
        # NAF like (mean(abs(log mag stft 1 - log mag stft 2))
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

def compute_t60(true_in, gen_in, fs):
    ch = true_in.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        try:
            true = pyroomacoustics.experimental.measure_rt60(true_in[c], fs=fs, decay_db=30)
            gen = pyroomacoustics.experimental.measure_rt60(gen_in[c], fs=fs, decay_db=30)
        except:
            true = -1
            gen = -1
        gt.append(true)
        pred.append(gen)
    return np.array(gt), np.array(pred)

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

def energy_decay(pred_mag, gt_mag):
    pred_mag = torch.from_numpy(pred_mag)
    gt_mag = torch.from_numpy(gt_mag)
    gt_mag = torch.sum(gt_mag ** 2, dim=1) #2 if batch
    pred_mag = torch.sum(pred_mag ** 2, dim=1) #2 if batch
    gt_mag = torch.log1p(gt_mag)
    pred_mag = torch.log1p(pred_mag)
    loss = F.l1_loss(gt_mag, pred_mag)
    return loss

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

