import numpy as np
import torch
import torch.nn as nn


class ParametricEqualizer(nn.Module):
    """Fast-parametric equalizer for approximation of Biquad IIR filter.
    """
    def __init__(self, sr: int, windows: int):
        """Initializer.
        Args:
            sr: sample rate.
            windows: size of the fft window.
        """
        super().__init__()
        self.sr = sr
        self.windows = windows

    def biquad(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Construct frequency level biquad filter.
        Args:
            a: [torch.float32; [..., 3]], recursive filter, iir.
            b: [torch.float32; [..., 3]], finite impulse filter.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], biquad filter.
        """
        iir = torch.fft.rfft(a, self.windows, dim=-1)
        fir = torch.fft.rfft(b, self.windows, dim=-1)
        return fir / iir

    def low_shelving(self,
                     cutoff: float,
                     gain: torch.Tensor,
                     q: torch.Tensor) -> torch.Tensor:
        """Frequency level low-shelving filter.
        Args:
            cutoff: cutoff frequency.
            gain: [torch.float32; [...]], boost of attenutation in decibel.
            q: [torch.float32; [...]], quality factor.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], frequency filter.
        """
        # ref: torchaudio.functional.lowpass_biquad
        w0 = 2 * np.pi * cutoff / self.sr
        alpha = np.sin(w0) / 2.0 / q
        A = torch.exp(gain / 40 * np.log(10))
        temp1 = 2 * torch.sqrt(A) * alpha
        temp2 = (A - 1) * np.cos(w0)
        temp3 = (A + 1) * np.cos(w0)
        # [...], fir
        b0 = A * ((A + 1) - temp2 + temp1) # A * ((A+1) - ((A-1)*torch.cos(w0)) + (2 * sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - temp3)
        b2 = A * ((A + 1) - temp2 - temp1)
        # [...], iir
        a0 = (A + 1) + temp2 + temp1
        a1 = -2 * ((A - 1) + temp3)
        a2 = (A + 1) + temp2 - temp1

        return self.biquad(a=torch.stack([a0, a1, a2], dim=-1),
                      b=torch.stack([b0, b1, b2], dim=-1))

    def high_shelving(self,
                      cutoff: float,
                      gain: torch.Tensor,
                      q: torch.Tensor) -> torch.Tensor:
        """Frequency level high-shelving filter.
        Args:
            cutoff: cutoff frequency.
            gain: [torch.float32; [...]], boost of attenutation in decibel.
            q: [torch.float32; [...]], quality factor.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], frequency filter.
        """
        # ref: torchaudio.functional.highpass_biquad
        w0 = 2 * np.pi * cutoff / self.sr
        alpha = np.sin(w0) / 2.0 / q
        A = torch.exp(gain / 40 * np.log(10))

        temp1 = 2 * torch.sqrt(A) * alpha
        temp2 = (A - 1) * np.cos(w0)
        temp3 = (A + 1) * np.cos(w0)

        b0 = A * ((A + 1) + temp2 + temp1)
        b1 = -2 * A * ((A - 1) + temp3)
        b2 = A * ((A + 1) + temp2 - temp1)
        a0 = (A + 1) - temp2 + temp1
        a1 = 2 * ((A - 1) - temp3)
        a2 = (A + 1) - temp2 - temp1

        return self.biquad(a=torch.stack([a0, a1, a2], dim=-1),
                           b=torch.stack([b0, b1, b2], dim=-1))

    def peaking_equalizer(self,
                          center: torch.Tensor,
                          gain: torch.Tensor,
                          q: torch.Tensor) -> torch.Tensor:
        """Frequency level peaking equalizer.
        Args:
            center: [torch.float32; [...]], center frequency.
            gain: [torch.float32; [...]], boost or attenuation in decibel.
            q: [torch.float32; [...]], quality factor.
        Returns:
            [torch.float32; [..., windows // 2 + 1]], frequency filter.
        """
        # ref: torchaudio.functional.highpass_biquad
        # [...]
        # w0 = 2 * np.pi * center / self.sr
        # # [...]
        # alpha = torch.sin(w0) / 2 / q
        # cos_w0 = torch.cos(w0)
        # A = (gain / 40. * np.log(10)).exp()
        # # [..., windows // 2 + 1]
        # return self.biquad(
        #     a=torch.stack([1 + alpha / A, -2 * cos_w0, 1 - alpha / A], dim=-1),
        #     b=torch.stack([1 + alpha * A, -2 * cos_w0, 1 - alpha * A], dim=-1))
    
        w0 = 2 * np.pi * center / self.sr
        A = torch.exp(gain / 40.0 * np.log(10))
        alpha = torch.sin(w0) / 2 / q
        
        b0 = 1 + alpha * A
        b1 = -2 * torch.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha / A
        return self.biquad(a=torch.stack([a0, a1, a2], dim=-1),
                      b=torch.stack([b0, b1, b2], dim=-1))
