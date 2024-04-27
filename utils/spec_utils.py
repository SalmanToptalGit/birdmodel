import torch
import torch.nn as nn
from nnAudio.features.stft import STFT as nnAudioSTFT
from typing import Optional
from torchaudio.transforms import MelScale
EPSILON_FP16 = 1e-5
import numpy as np
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6, exportable=False):
        super().__init__()
        self.eps = eps
        self.exportable = exportable

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        if self.exportable:
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        else:
            norm_min, norm_max = (
                Xstd.min(-1)[0].min(-1)[0],
                Xstd.max(-1)[0].max(-1)[0],
            )
            fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
                (norm_max - norm_min)
            )
            V = torch.zeros_like(Xstd)
            if fix_ind.sum():
                V_fix = Xstd[fix_ind]
                norm_max_fix = norm_max[fix_ind, None, None]
                norm_min_fix = norm_min[fix_ind, None, None]
                V_fix = torch.max(
                    torch.min(V_fix, norm_max_fix),
                    norm_min_fix,
                )
                V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
                V[fix_ind] = V_fix
            return V


class TraceableMelspec(nn.Module):
    def __init__(
        self,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        # Mel params
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_fft: int = 400,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        # Add params
        trainable: bool = False,
        quantizable: bool = False,
    ):
        super().__init__()
        self.spectrogram = nnAudioSTFT(
                n_fft=n_fft,
                win_length=win_length,
                freq_bins=None,
                hop_length=hop_length,
                window="hann",
                freq_scale="no",
                # Do not define `fmin` and `fmax`, because freq_scale = "no"
                center=center,
                pad_mode=pad_mode,
                iSTFT=False,
                sr=sample_rate,
                trainable=trainable,
                output_format="Complex",
                verbose=True,
            )
        self.normalized = normalized
        self.power = power
        self.register_buffer(
            "window",
            torch.hann_window(win_length if win_length is not None else n_fft),
        )
        self.trainable = trainable
        self.mel_scale = MelScale(
            n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, x):
        spec_f = self.spectrogram(x)
        if self.normalized:
            spec_f /= self.window.pow(2.0).sum().sqrt()
        if self.power is not None:
            # prevent Nan gradient when sqrt(0) due to output=0
            # Taken from nnAudio.features.stft.STFT
            eps = 1e-8 if self.trainable else 0.0
            spec_f = torch.sqrt(
                spec_f[:, :, :, 0].pow(2) + spec_f[:, :, :, 1].pow(2) + eps
            )
            if self.power != 1.0:
                spec_f = spec_f.pow(self.power)
        mel_spec = self.mel_scale(spec_f)
        return mel_spec



class CustomMasking(nn.Module):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(
                    low=1, high=self.mask_max_masks + 1
                )
                for _ in range(n_applies):
                    if self.inplace:
                        x[i : i + 1] = self.mask_module(x[i : i + 1])
                    else:
                        output[i : i + 1] = self.mask_module(output[i : i + 1])
        if self.inplace:
            return x
        else:
            return output


class CustomTimeMasking(CustomMasking):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
            inplace=inplace,
        )
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
            inplace=inplace,
        )
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_config, time_mask_config):

        super(SpecAugment, self).__init__()
        self.spec_augment = []
        self.spec_augment.append(
            CustomFreqMasking(**freq_mask_config)
        )
        self.spec_augment.append(
            CustomTimeMasking(**time_mask_config)
        )
        self.spec_augment = nn.Sequential(*self.spec_augment)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spec_augment(x)

class SpecFeatureExtractor(nn.Module):
    def __init__(self, mel_spec_paramms, exportable, top_db):

        super(SpecFeatureExtractor, self).__init__()
        self.logmelspec_extractor = self._create_feature_extractor(mel_spec_paramms, exportable, top_db, False, "Melspec")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.logmelspec_extractor(x)[:, None]
        return spec


    def _create_feature_extractor(
            self, mel_spec_paramms, exportable, top_db, quantizable, spec_extractor
    ):
        if spec_extractor == "Melspec":
            if exportable:
                spec_init = TraceableMelspec
            else:
                spec_init = MelSpectrogram
        else:
            raise NotImplementedError(f"{spec_extractor} not implemented")

        self._n_specs = 1
        return nn.Sequential(
            spec_init(**mel_spec_paramms, quantizable=True)
            if quantizable
            else spec_init(**mel_spec_paramms),
            AmplitudeToDB(top_db=top_db),
            NormalizeMelSpec(exportable=exportable),
        )
