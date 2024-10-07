import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from NeRAF.NeRAF_helper import SpectralLoss, compute_t60, evaluate_edt, evaluate_clarity, Envelope_distance

""" Losses """
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

"""
class NACF_decayLoss(nn.Module):
    def __init__(self):
        super(NACF_decayLoss, self).__init__()
    
    def forward(self, x_mag, y_mag):
        # sum on frequency axis 
        energy_x = torch.sum(x_mag ** 2, dim=2) 
        energy_y = torch.sum(y_mag ** 2, dim=2)

        # use cumsum to do the same
        decay_x = torch.flip(torch.cumsum(torch.flip(energy_x, [-1]), dim=2),[-1])
        decay_y = torch.flip(torch.cumsum(torch.flip(energy_y, [-1]), dim=2), [-1])

        decay_curve_x = torch.log10(decay_x + 1e-13)
        decay_curve_y = torch.log10(decay_y + 1e-13)

        loss = F.l1_loss(decay_curve_x, decay_curve_y)
        return loss
"""

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, loss_type = 'l1'
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(loss_type=loss_type)
        #self.self.energy_decay_loss = NACF_decayLoss() # Can be use only on FULL STFT 

    def forward(self, x_log, y_log):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        # From log to mag STFT
        x_mag = torch.exp(x_log) - 1e-3
        y_mag = torch.exp(y_log) - 1e-3
        # apply losses
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_log, y_log)
        #energy_decay_loss = self.energy_decay_loss(x_mag, y

        return {'audio_sc_loss': sc_loss, 'audio_mag_loss': mag_loss}

""" RAF Evaluator """
class RAFEvaluator(object):
    def __init__(self, fs=48000):
        self.fs = fs
        self.spectral_loss_mag = SpectralLoss(base_loss=F.l1_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='mag')
        self.spectral_loss_logmag = SpectralLoss(base_loss=F.l1_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='log mag')

        if self.fs == 48000:
            self.n_fft = 1024
            self.win_length = 512
            self.hop_len = 256
        elif self.fs == 16000:
            self.n_fft = 512
            self.win_length = 256
            self.hop_len = 128
        else: 
            raise ValueError('Sample rate not supported')
        
        # To assure consistency with RAF benchmark, we go back to STFT from our waveform
        self.transform_stft_torch = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_len, power=None)
    
    def get_full_metrics(self, mag_prd, mag_gt, wav_gt_ff, wav_pred_istft, wav_gt_istft,  log_prd, log_gt):#, gl=False):
        
        # Get waveform from mag STFT
        wav_prd = wav_pred_istft
        wav_gt = wav_gt_istft
        n_ch = wav_gt.shape[0]

        # zero pad waveform to be the same size as gt ff (i.e. max_len * hop_len )
        wav_prd = np.pad(wav_prd, ((0,0),(0, wav_gt_ff.shape[1]-wav_prd.shape[1])), 'constant', constant_values=(0,0))
        wav_gt = np.pad(wav_gt, ((0,0),(0, wav_gt_ff.shape[1]-wav_gt.shape[1])), 'constant', constant_values=(0,0))

        ## STFT related
        # Compute RAF spectral loss by going from STFT to waveform and then back to STFT
        mag_prd_from_istft = self.transform_stft_torch(torch.tensor(wav_prd))
        log_prd_from_istft = torch.log(torch.abs(mag_prd_from_istft) + 1e-3)
        log_prd_from_istft = log_prd_from_istft[..., :log_gt.shape[2]].numpy()
        RAF_spectral = self.spectral_loss_logmag(torch.from_numpy(log_prd_from_istft), torch.from_numpy(log_gt))
        
        ## Waveform related against GT from file

        # T60 advanced to match RAF benchmark
        t60s_gt, t60s_prd = compute_t60(wav_gt_ff, wav_prd, fs=self.fs, advanced = True)
        t60s = np.concatenate((t60s_gt, t60s_prd))
        t60s = np.expand_dims(t60s, axis=0)
        diff = np.abs(t60s[:,n_ch:]-t60s[:,:n_ch])/np.abs(t60s[:,:n_ch])
        mask = np.any(t60s<-0.5, axis=1)
        diff = np.mean(diff, axis=1)
        diff[mask] = 1
        mean_t60error_gt_ff_advanced = np.mean(diff)*100
        invalid_advanced = np.sum(mask)

        # EDT
        edt_gt, edt_prd = evaluate_edt(wav_prd, wav_gt_ff, fs=self.fs)
        edts = np.concatenate((edt_gt, edt_prd))
        edt_instance = np.abs(edts[n_ch:]-edts[:n_ch]) # pred-gt
        mean_edt = np.mean(edt_instance, axis=0) # mean over instance channels

        # C50
        c50_gt, c50_prd = evaluate_clarity(wav_prd, wav_gt_ff, fs=self.fs)
        c50s = np.concatenate((c50_gt, c50_prd))
        c50_instance = np.abs(c50s[n_ch:]-c50s[:n_ch]) # pred-gt
        mean_c50 = np.mean(c50_instance, axis=0) # mean over instance channels

        res = {
                "audio_T60": mean_t60error_gt_ff_advanced,
                "audio_total_invalids_T60": invalid_advanced,
                "audio_stft_error": RAF_spectral,
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
    
    def get_stft_metrics(self,mag_prd,mag_gt,gl=False):
        ## STFT related
        mag_loss = torch.mean(torch.pow(mag_prd - mag_gt, 2)) * 2
        spec_loss = self.spectral_loss_mag(mag_prd, mag_gt).item()

        return {
                "audio_mag": mag_loss,
                "audio_spectral_loss": spec_loss,
        }
        
""" Sound Spaces Evaluator """
class SoundSpacesEvaluator(object):
    def __init__(self, fs=22050):
        self.fs = fs # sample rate
    
    def get_full_metrics(self, mag_prd, mag_gt, wav_gt_ff, wav_pred_istft, wav_gt_istft,  log_prd, log_gt):#, gl=False):
        
        wav_prd = wav_pred_istft
        wav_gt = wav_gt_istft
        n_ch = wav_gt.shape[0]

        # Zero pad waveform to be the same size as gt ff (i.e., max_len * hop_len )
        wav_prd = np.pad(wav_prd, ((0,0),(0, wav_gt_ff.shape[1]-wav_prd.shape[1])), 'constant', constant_values=(0,0))
        wav_gt = np.pad(wav_gt, ((0,0),(0, wav_gt_ff.shape[1]-wav_gt.shape[1])), 'constant', constant_values=(0,0))

        ## Waveform related
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
                "audio_T60_mean_error": mean_t60error,
                "audio_total_invalids_T60": invalid,
                "audio_EDT": mean_edt,
                "audio_C50": mean_c50,
                }
        
        for key in res.keys():
            #if tensor go to numpy
            if isinstance(res[key], torch.Tensor):
                res[key] = res[key].item()
            else:
                res[key] = float(res[key])

        return res
        
    
    def get_stft_metrics(self,mag_prd,mag_gt):
        ## STFT related
        mag_loss = torch.mean(torch.pow(mag_prd - mag_gt, 2)) * 2

        return {
                "audio_mag": mag_loss,
        }
    
