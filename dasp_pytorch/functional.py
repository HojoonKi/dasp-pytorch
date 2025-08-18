import torch
import numpy as np
import scipy.signal
import dasp_pytorch.signal
import torchaudio.functional as TAF

from functools import partial
from typing import Dict, List, Union
# change?

def gain(x: torch.Tensor, sample_rate: int, gain_db: torch.Tensor):
    """Apply gain in dB to audio tensor.

    The same gain will be applied to each audio channel in the tensor.

    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (int): Audio sample rate.
        gain_db (torch.Tensor): Gain in dB with shape (bs)

    Returns:
        torch.Tensor: Output audio tensor with shape (bs, chs, seq_len)

    """
    bs, chs, seq_len = x.size()

    gain_db = gain_db.view(bs, 1, 1)
    # convert gain from db to linear
    gain_lin = 10 ** (gain_db.repeat(1, chs, 1) / 20.0)
    return x * gain_lin


def stereo_bus(x: torch.tensor, sample_rate: int, send_db: torch.Tensor):
    """Stereo bus with controllable send levels.

    Given a tensor containing a set of audio tracks, create a stereo bus by summing the tracks
    according to the relative send levels. The result will be a stereo signal. Note that all
    input tracks are expected to be stereo.

    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, 2, tracks, seq_len)
        sample_rate (int): Audio sample rate.
        send_db (torch.Tensor): Send levels in dB with shape (bs, tracks, 1)

    Returns:
        torch.Tensor: Output audio tensor with shape (bs, 2, seq_len)

    """
    bs, chs, tracks, seq_len = x.size()

    assert chs == 2, "Input tensor must have shape (bs, 2, tracks, seq_len)"

    # convert send levels from db to linear
    sends_lin = 10 ** (send_db.view(bs, 1, tracks, 1) / 20.0)

    # apply send levels
    x_bus = x * sends_lin

    # sum tracks to stereo bus
    x_bus = x_bus.sum(dim=2, keepdim=False)

    # apply send levels
    return x_bus


def distortion(x: torch.Tensor, sample_rate: int, drive_db: torch.Tensor):
    """Simple soft-clipping distortion with drive control.

    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, chs, seq_len)
        sample_rate (int): Audio sample rate.
        drive_db (torch.Tensor): Drive in dB with shape (bs)

    Returns:
        torch.Tensor: Output audio tensor with shape (bs, chs, seq_len)

    """
    bs, chs, seq_len = x.size()
    return torch.tanh(x * (10 ** (drive_db.view(bs, chs, -1) / 20.0)))


def advanced_distortion(
    x: torch.Tensor,
    sample_rate: float,
    input_gain_db: torch.Tensor,
    output_gain_db: torch.Tensor,
    tone: torch.Tensor,
    dc_offset: torch.Tensor,
):
    """

    Args:
        x (torch.Tensor): Input audio tensor with shape (bs, ..., seq_len)
        sample_rate (float): Audio sample rate.
        input_gain_db (torch.Tensor): Input gain in dB with shape (bs, ...)
        output_gain_db (torch.Tensor): Output gain in dB with shape (bs, ...)
        tone (torch.Tensor): Tone control with shape (bs, ...)
        dc_offset (torch.Tensor): DC offset with shape (bs, ...)

    The tone filter is implemented as a weighted sum of 1st order highpass and lowpass filters.
    This is based on the design of the Boss guituar pedal modelled in [2]. Their design uses a
    highpass with corner frequency 1.16 kHz and a lowpass with corner frequency 320 Hz.

    [1] Colonel, Joseph T., Marco Comunità, and Joshua Reiss.
        "Reverse Engineering Memoryless Distortion Effects with Differentiable Waveshapers."
        153rd Convention of the Audio Engineering Society. Audio Engineering Society, 2022.

    [2] Yeh, David Te-Mao.
        Digital implementation of musical distortion circuits by analysis and simulation.
        Stanford University, 2009.
    """
    raise NotImplementedError

def mean_distortion(audio: torch.Tensor, params: Dict) -> torch.Tensor:
        drive_db = params['gain'].view(-1, 1, 1).clamp(0.0, 15.0)
        color = params['color'].view(-1, 1, 1).clamp(-0.9, 0.9)
        
        # 1. 입력 오디오의 AC RMS 계산 (DC 제거)
        input_mean = torch.mean(audio, dim=-1, keepdim=True)
        input_ac = audio - input_mean
        input_rms = torch.sqrt(torch.mean(input_ac**2, dim=-1, keepdim=True) + 1e-8)
        
        # 2. dB 기반 드라이브 적용
        linear_gain = 10 ** (drive_db / 20.0)
        distorted_audio = torch.tanh((audio + color) * linear_gain)
        
        # 3. 왜곡된 오디오의 AC RMS 계산 (DC 제거)
        output_mean = torch.mean(distorted_audio, dim=-1, keepdim=True)
        output_ac = distorted_audio - output_mean
        output_rms = torch.sqrt(torch.mean(output_ac**2, dim=-1, keepdim=True) + 1e-8)
        
        # 4. 출력 AC RMS를 입력 AC RMS와 맞추도록 스케일링 (0 나누기 방지)
        scale = input_rms / output_rms.clamp(min=1e-8)
        normalized_audio = output_ac * scale
        normalized_audio = 0.7 * normalized_audio + 0.3 * audio
        
        # 옵션: 원본 input_mean을 추가할 수 있지만, DC 제거를 위해 0으로 유지 추천
        # normalized_audio += input_mean  # 필요 시 활성화
        
        return normalized_audio


def graphic_eq(x: torch.Tensor, sample_rate: float):
    raise NotImplementedError


def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    low_shelf_gain_db: torch.Tensor,
    low_shelf_cutoff_freq: torch.Tensor,
    low_shelf_q_factor: torch.Tensor,
    band0_gain_db: torch.Tensor,
    band0_cutoff_freq: torch.Tensor,
    band0_q_factor: torch.Tensor,
    band1_gain_db: torch.Tensor,
    band1_cutoff_freq: torch.Tensor,
    band1_q_factor: torch.Tensor,
    band2_gain_db: torch.Tensor,
    band2_cutoff_freq: torch.Tensor,
    band2_q_factor: torch.Tensor,
    band3_gain_db: torch.Tensor,
    band3_cutoff_freq: torch.Tensor,
    band3_q_factor: torch.Tensor,
    high_shelf_gain_db: torch.Tensor,
    high_shelf_cutoff_freq: torch.Tensor,
    high_shelf_q_factor: torch.Tensor,
):
    """Six-band Parametric Equalizer.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    [1] Välimäki, Vesa, and Joshua D. Reiss.
        "All about audio equalization: Solutions and frontiers."
        Applied Sciences 6.5 (2016): 129.

    [2] Nercessian, Shahan.
        "Neural parametric equalizer matching using differentiable biquads."
        Proc. Int. Conf. Digital Audio Effects (eDAFx-20). 2020.

    [3] Colonel, Joseph T., Christian J. Steinmetz, et al.
        "Direct design of biquad filter cascades with deep learning by sampling random polynomials."
        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

    [4] Steinmetz, Christian J., Nicholas J. Bryan, and Joshua D. Reiss.
        "Style Transfer of Audio Effects with Differentiable Signal Processing."
        Journal of the Audio Engineering Society. Vol. 70, Issue 9, 2022, pp. 708-721.

    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        low_shelf_gain_db (torch.Tensor): Low-shelf filter gain in dB.
        low_shelf_cutoff_freq (torch.Tensor): Low-shelf filter cutoff frequency in Hz.
        low_shelf_q_factor (torch.Tensor): Low-shelf filter Q-factor.
        band0_gain_db (torch.Tensor): Band 1 filter gain in dB.
        band0_cutoff_freq (torch.Tensor): Band 1 filter cutoff frequency in Hz.
        band0_q_factor (torch.Tensor): Band 1 filter Q-factor.
        band1_gain_db (torch.Tensor): Band 2 filter gain in dB.
        band1_cutoff_freq (torch.Tensor): Band 2 filter cutoff frequency in Hz.
        band1_q_factor (torch.Tensor): Band 2 filter Q-factor.
        band2_gain_db (torch.Tensor): Band 3 filter gain in dB.
        band2_cutoff_freq (torch.Tensor): Band 3 filter cutoff frequency in Hz.
        band2_q_factor (torch.Tensor): Band 3 filter Q-factor.
        band3_gain_db (torch.Tensor): Band 4 filter gain in dB.
        band3_cutoff_freq (torch.Tensor): Band 4 filter cutoff frequency in Hz.
        band3_q_factor (torch.Tensor): Band 4 filter Q-factor.
        high_shelf_gain_db (torch.Tensor): High-shelf filter gain in dB.
        high_shelf_cutoff_freq (torch.Tensor): High-shelf filter cutoff frequency in Hz.
        high_shelf_q_factor (torch.Tensor): High-shelf filter Q-factor.

    Returns:
        y (torch.Tensor): Filtered signal.
    """
    bs, chs, seq_len = x.size()

    # reshape to move everything to batch dim
    # x = x.view(-1, 1, seq_len)
    low_shelf_gain_db = low_shelf_gain_db.view(-1, 1, 1)
    low_shelf_cutoff_freq = low_shelf_cutoff_freq.view(-1, 1, 1)
    low_shelf_q_factor = low_shelf_q_factor.view(-1, 1, 1)
    band0_gain_db = band0_gain_db.view(-1, 1, 1)
    band0_cutoff_freq = band0_cutoff_freq.view(-1, 1, 1)
    band0_q_factor = band0_q_factor.view(-1, 1, 1)
    band1_gain_db = band1_gain_db.view(-1, 1, 1)
    band1_cutoff_freq = band1_cutoff_freq.view(-1, 1, 1)
    band1_q_factor = band1_q_factor.view(-1, 1, 1)
    band2_gain_db = band2_gain_db.view(-1, 1, 1)
    band2_cutoff_freq = band2_cutoff_freq.view(-1, 1, 1)
    band2_q_factor = band2_q_factor.view(-1, 1, 1)
    band3_gain_db = band3_gain_db.view(-1, 1, 1)
    band3_cutoff_freq = band3_cutoff_freq.view(-1, 1, 1)
    band3_q_factor = band3_q_factor.view(-1, 1, 1)
    high_shelf_gain_db = high_shelf_gain_db.view(-1, 1, 1)
    high_shelf_cutoff_freq = high_shelf_cutoff_freq.view(-1, 1, 1)
    high_shelf_q_factor = high_shelf_q_factor.view(-1, 1, 1)

    eff_bs = x.size(0)

    # six second order sections
    sos = torch.zeros(eff_bs, 6, 6).type_as(low_shelf_gain_db)
    # ------------ low shelf ------------
    b, a = dasp_pytorch.signal.biquad(
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    sos[:, 0, :] = torch.cat((b, a), dim=-1)
    # ------------ band0 ------------
    b, a = dasp_pytorch.signal.biquad(
        band0_gain_db,
        band0_cutoff_freq,
        band0_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 1, :] = torch.cat((b, a), dim=-1)
    # ------------ band1 ------------
    b, a = dasp_pytorch.signal.biquad(
        band1_gain_db,
        band1_cutoff_freq,
        band1_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 2, :] = torch.cat((b, a), dim=-1)
    # ------------ band2 ------------
    b, a = dasp_pytorch.signal.biquad(
        band2_gain_db,
        band2_cutoff_freq,
        band2_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 3, :] = torch.cat((b, a), dim=-1)
    # ------------ band3 ------------
    b, a = dasp_pytorch.signal.biquad(
        band3_gain_db,
        band3_cutoff_freq,
        band3_q_factor,
        sample_rate,
        "peaking",
    )
    sos[:, 4, :] = torch.cat((b, a), dim=-1)
    # ------------ high shelf ------------
    b, a = dasp_pytorch.signal.biquad(
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos[:, 5, :] = torch.cat((b, a), dim=-1)

    x_out = dasp_pytorch.signal.sosfilt_via_fsm(sos, x)

    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out

import matplotlib.pyplot as plt
def visualize_filter_response(sos_matrix, sample_rate):
    """
    주어진 SOS 행렬의 주파수 응답을 시각화합니다. (버그 수정됨)
    """
    print("🔬 Visualizing filter frequency response...")
    if sos_matrix.is_cuda:
        sos_matrix = sos_matrix.cpu()

    sos_single = sos_matrix[0].detach().numpy() 
    
    n_fft = 4096
    H_total = np.ones(n_fft // 2 + 1, dtype=np.complex64) # length: 2049
    
    for section in sos_single:
        b, a = section[:3], section[3:]
        print(b, a)
        w, h = scipy.signal.freqz(b, a, worN=n_fft // 2 + 1, fs=sample_rate)
        
        H_total *= h

    plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(np.abs(H_total) + 1e-8))
    plt.title("Frequency Response of the Generated EQ Filter")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.xscale('log')
    plt.xlim([20, sample_rate / 2])
    plt.ylim([-40, 10])
    plt.savefig("filter_response.png") 
    print("   -> Saved 'filter_response.png'")

def differentiable_flexible_eq(
    x: torch.Tensor,
    sample_rate: float,
    bands_params: Dict[str, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    Applies 5 bands of EQ with differentiable filter type selection,
    accepting parameters as a dictionary of bands.
    """
    num_bands = len(bands_params)
    bs, chs, seq_len = x.size()
    
    filter_types = ["low_shelf", "peaking", "high_shelf", "high_pass", "low_pass"]
    num_filter_types = len(filter_types)
    
    final_sos = torch.zeros(bs, num_bands, 6, device=x.device, dtype=x.dtype)

    for i, band_name in enumerate(sorted(bands_params.keys())):
        band_p = bands_params[band_name]
        
        filter_choice_one_hot = torch.nn.functional.gumbel_softmax(band_p['filter_type'], tau=1.0, hard=True)
        all_coeffs = []
        for f_type in filter_types:
            b, a = dasp_pytorch.signal.biquad(
                band_p['gain_db'], band_p['center_freq'], band_p['q'],
                sample_rate, f_type
            )
            all_coeffs.append(torch.cat((b, a), dim=-1))
        
        all_coeffs = torch.stack(all_coeffs, dim=0).permute(1, 0, 2)
        
        selected_sos = torch.bmm(filter_choice_one_hot.unsqueeze(1), all_coeffs).squeeze(1)
        final_sos[:, i, :] = selected_sos
        
    x_reshaped = x.view(bs * chs, -1)
    sos_expanded = final_sos.repeat_interleave(chs, dim=0)
    visualize_filter_response(sos_expanded, sample_rate)
    y_reshaped = dasp_pytorch.signal.sosfilt_via_fsm(sos_expanded, x_reshaped)
    
    y = y_reshaped.view(bs, chs, seq_len)
    
    return y

def differentiable_hybrid_eq( # 함수 이름 변경
    x: torch.Tensor,
    sample_rate: float,
    bands_params: Dict[str, Dict[str, torch.Tensor]],
    training: bool,
) -> torch.Tensor:
    """
    Applies a 6-band hybrid EQ with constrained filter types for stable training.
    """
    bs, chs, seq_len = x.size()
    num_bands = 6  # 고정 6개 밴드
    final_sos = torch.zeros(bs, num_bands, 6, device=x.device, dtype=x.dtype)

    for i, band_name in enumerate(sorted(bands_params.keys())):
        band_p = bands_params[band_name]
        gain, freq, q = band_p['gain_db'], band_p['center_freq'], band_p['q']
        # 밴드 인덱스에 따라 필터 타입 후보군을 다르게 설정
        if i == 0: # Band 1
            filter_types = ["high_pass", "low_shelf"]
        elif i == num_bands - 1: # Last Band
            filter_types = ["low_pass", "high_shelf"]
        else: # Middle Bands
            filter_types = ["peaking"]

        # 모든 가능한 필터 타입에 대한 계수를 계산
        all_coeffs = []
        for f_type in filter_types:
            b, a = dasp_pytorch.signal.biquad(gain, freq, q, sample_rate, f_type)
            all_coeffs.append(torch.cat((b, a), dim=-1))
        
        all_coeffs = torch.stack(all_coeffs, dim=0).permute(1, 0, 2)  # (batch_size, num_filter_types, 6)
        if len(filter_types) > 1:
            logits = band_p['filter_type']
            if training:
                # 학습 시: Gumbel-Softmax로 미분 가능한 확률적 샘플링
                filter_choice_one_hot = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)
            else:
                # 평가 시: 가장 확률 높은 필터를 확정적으로 선택
                indices = torch.argmax(logits, dim=-1)
                filter_choice_one_hot = torch.nn.functional.one_hot(indices, num_classes=logits.shape[-1]).float()
        else:
            filter_choice_one_hot = torch.ones(bs, 1, device=x.device)
        # 선택된 필터 계수 계산
        selected_sos = torch.bmm(filter_choice_one_hot.unsqueeze(1), all_coeffs).squeeze(1)  # (batch_size, 6)
        final_sos[:, i, :] = selected_sos
    x_reshaped = x.view(bs * chs, -1)
    sos_expanded = final_sos.repeat_interleave(chs, dim=0)
    visualize_filter_response(sos_expanded, sample_rate)
    y_reshaped = dasp_pytorch.signal.sosfilt_via_fsm(sos_expanded, x_reshaped)
    
    return y_reshaped.view(bs, chs, seq_len)

def compressor(
    x: torch.Tensor,
    sample_rate: float,
    threshold_db: torch.Tensor,
    ratio: torch.Tensor,
    attack_ms: torch.Tensor,
    release_ms: torch.Tensor,
    knee_db: torch.Tensor,
    makeup_gain_db: torch.Tensor,
    eps: float = 1e-8,
    lookahead_samples: int = 0,
):
    """Dynamic range compressor.

    This compressor is based on the standard feedforward digital compressor design [1].
    To make the implementation differentiable the the simple compressor proposed in [2] is used.
    However, the original design utilized a single time constant for the attack and release
    ballisitics in order to parallelize the branching recursive smoothing filter.
    This implementation builds on this using approximate ballisitics similar to those
    introduced in [3]. This involves applying two 1-pole lowpass filters to the gain reduction curve,
    each with different time constants. The final smoothed gain reduction curve is then computed
    by combining these two smoothed curves, setting only one to be active at a time based on
    the level of the gain reduction curve in relation to the threshold.

    [1] Giannoulis, Dimitrios, Michael Massberg, and Joshua D. Reiss.
        "Digital Dynamic Range Compressor Design - A Tutorial and Analysis."
        Journal of Audio Engineering Society. Vol. 60, Issue 6, 2012, pp. 399-408.

    [2] Steinmetz, Christian J., Nicholas J. Bryan, and Joshua D. Reiss.
        "Style Transfer of Audio Effects with Differentiable Signal Processing."
        Journal of the Audio Engineering Society. Vol. 70, Issue 9, 2022, pp. 708-721.

    [3] Colonel, Joseph, and Joshua D. Reiss.
        "Approximating Ballistics in a Differentiable Dynamic Range Compressor."
        153rd Convention of the Audio Engineering Society. Audio Engineering Society, 2022.

    Args:
        x (torch.Tensor): Audio tensor with shape (bs, chs, seq_len).
        sample_rate (float): Audio sampling rate.
        threshold_db (torch.Tensor): Threshold at which to begin gain reduction.
        ratio (torch.Tensor): Amount to reduce gain as a function of the distance above threshold.
        attack_ms (torch.Tensor): Attack time in milliseconds.
        release_ms (torch.Tensor): Release time in milliseconds.
        knee_db (torch.Tensor): Softness of the knee. Higher = softer. (Must be positive)
        makeup_gain_db (torch.Tensor): Apply gain after compression to restore signal level.
        eps (float): Epsilon value for numerical stability. Default: 1e-8

    Returns:
        y (torch.Tensor): Compressed signal.
    """
    bs, chs, seq_len = x.size()  # check shape

    # if multiple channels are present create sum side-chain
    x_side = x.sum(dim=1, keepdim=True)
    x_side = x_side.view(-1, 1, seq_len)
    threshold_db = threshold_db.view(-1, 1, 1)
    ratio = ratio.view(-1, 1, 1)
    attack_ms = attack_ms.view(-1, 1, 1)
    release_ms = release_ms.view(-1, 1, 1)
    knee_db = knee_db.view(-1, 1, 1)
    makeup_gain_db = makeup_gain_db.view(-1, 1, 1)
    eff_bs = x_side.size(0)

    # compute time constants
    normalized_attack_time = sample_rate * (attack_ms / 1e3)
    # normalized_release_time = sample_rate * (release_ms / 1e3)
    constant = torch.tensor([9.0]).type_as(attack_ms)
    alpha_A = torch.exp(-torch.log(constant) / normalized_attack_time)
    # alpha_R = torch.exp(-torch.log(constant) / normalized_release_time)
    # note that release time constant is not used in the smoothing filter

    # compute energy in db
    x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))

    # static characteristic with soft knee
    x_sc = x_db.clone()

    # when signal is less than (T - W/2) leave as x_db

    # when signal is at the threshold engage knee
    idx1 = x_db >= (threshold_db - (knee_db / 2))
    idx2 = x_db <= (threshold_db + (knee_db / 2))
    idx = torch.logical_and(idx1, idx2)
    x_sc_below = x_db + ((1 / ratio) - 1) * (
        (x_db - threshold_db + (knee_db / 2)) ** 2
    ) / (2 * knee_db)
    x_sc[idx] = x_sc_below[idx]

    # when signal is above threshold linear response
    idx = x_db > (threshold_db + (knee_db / 2))
    x_sc_above = threshold_db + ((x_db - threshold_db) / ratio)
    x_sc[idx] = x_sc_above[idx]

    # output of gain computer
    g_c = x_sc - x_db

    # design attack/release smoothing filter
    b = torch.cat(
        [(1 - alpha_A), torch.zeros(eff_bs, 1, 1).type_as(alpha_A)],
        dim=-1,
    ).squeeze(1)
    a = torch.cat(
        [torch.ones(eff_bs, 1, 1).type_as(alpha_A), -alpha_A],
        dim=-1,
    ).squeeze(1)
    g_c_attack = dasp_pytorch.signal.lfilter_via_fsm(g_c, b, a)

    # look-ahead by delaying the input signal in relation to gain reduction
    if lookahead_samples > 0:
        x = torch.roll(x, lookahead_samples, dims=-1)
        x[:, :, :lookahead_samples] = 0

    # add makeup gain in db
    g_s = g_c_attack + makeup_gain_db

    # convert db gains back to linear
    g_lin = 10 ** (g_s / 20.0)

    # apply time-varying gain and makeup gain
    y = x * g_lin

    # move channels back to the channel dimension
    y = y.view(bs, chs, seq_len)

    return y


def expander():
    raise NotImplementedError


def noise_shaped_reverberation(
    x: torch.Tensor,
    sample_rate: float,
    band0_gain: torch.Tensor,
    band1_gain: torch.Tensor,
    band2_gain: torch.Tensor,
    band3_gain: torch.Tensor,
    band4_gain: torch.Tensor,
    band5_gain: torch.Tensor,
    band6_gain: torch.Tensor,
    band7_gain: torch.Tensor,
    band8_gain: torch.Tensor,
    band9_gain: torch.Tensor,
    band10_gain: torch.Tensor,
    band11_gain: torch.Tensor,
    band0_decay: torch.Tensor,
    band1_decay: torch.Tensor,
    band2_decay: torch.Tensor,
    band3_decay: torch.Tensor,
    band4_decay: torch.Tensor,
    band5_decay: torch.Tensor,
    band6_decay: torch.Tensor,
    band7_decay: torch.Tensor,
    band8_decay: torch.Tensor,
    band9_decay: torch.Tensor,
    band10_decay: torch.Tensor,
    band11_decay: torch.Tensor,
    mix: torch.Tensor,
    num_samples: int = 65536,
    num_bandpass_taps: int = 1023,
):
    """Artificial reverberation using frequency-band noise shaping.

    This differentiable artificial reverberation model is based on the idea of
    filtered noise shaping, similar to that proposed in [1]. This approach leverages
    the well known idea that a room impulse response (RIR) can be modeled as the direct sound,
    a set of early reflections, and a decaying noise-like tail [2].

    [1] Steinmetz, Christian J., Vamsi Krishna Ithapu, and Paul Calamia.
        "Filtered noise shaping for time domain room impulse response estimation from reverberant speech."
        2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2021.

    [2] Moorer, James A.
        "About this reverberation business."
        Computer Music Journal (1979): 13-28.

    Args:
        x (torch.Tensor): Input audio signal. Shape (bs, chs, seq_len).
        sample_rate (float): Audio sample rate.
        band0_gain (torch.Tensor): Gain for first octave band on (0,1). Shape (bs, 1).
        band1_gain (torch.Tensor): Gain for second octave band on (0,1). Shape (bs, 1).
        band2_gain (torch.Tensor): Gain for third octave band on (0,1). Shape (bs, 1).
        band3_gain (torch.Tensor): Gain for fourth octave band on (0,1). Shape (bs, 1).
        band4_gain (torch.Tensor): Gain for fifth octave band on (0,1). Shape (bs, 1).
        band5_gain (torch.Tensor): Gain for sixth octave band on (0,1). Shape (bs, 1).
        band6_gain (torch.Tensor): Gain for seventh octave band on (0,1). Shape (bs, 1).
        band7_gain (torch.Tensor): Gain for eighth octave band on (0,1). Shape (bs, 1).
        band8_gain (torch.Tensor): Gain for ninth octave band on (0,1). Shape (bs, 1).
        band9_gain (torch.Tensor): Gain for tenth octave band on (0,1). Shape (bs, 1).
        band10_gain (torch.Tensor): Gain for eleventh octave band on (0,1). Shape (bs, 1).
        band11_gain (torch.Tensor): Gain for twelfth octave band on (0,1). Shape (bs, 1).
        band0_decays (torch.Tensor): Decay parameter for first octave band (0,1). Shape (bs, 1).
        band1_decays (torch.Tensor): Decay parameter for second octave band (0,1). Shape (bs, 1).
        band2_decays (torch.Tensor): Decay parameter for third octave band (0,1). Shape (bs, 1).
        band3_decays (torch.Tensor): Decay parameter for fourth octave band (0,1). Shape (bs, 1).
        band4_decays (torch.Tensor): Decay parameter for fifth octave band (0,1). Shape (bs, 1).
        band5_decays (torch.Tensor): Decay parameter for sixth octave band (0,1). Shape (bs, 1).
        band6_decays (torch.Tensor): Decay parameter for seventh octave band (0,1). Shape (bs, 1).
        band7_decays (torch.Tensor): Decay parameter for eighth octave band (0,1). Shape (bs, 1).
        band8_decays (torch.Tensor): Decay parameter for ninth octave band (0,1). Shape (bs, 1).
        band9_decays (torch.Tensor): Decay parameter for tenth octave band (0,1). Shape (bs, 1).
        band10_decays (torch.Tensor): Decay parameter for eleventh octave band (0,1). Shape (bs, 1).
        band11_decays (torch.Tensor): Decay parameter for twelfth octave band (0,1). Shape (bs, 1).
        mix (torch.Tensor): Mix between dry and wet signal. Shape (bs, 1).
        num_samples (int, optional): Number of samples to use for IR generation. Defaults to 88200.
        num_bandpass_taps (int, optional): Number of filter taps for the octave band filterbank filters. Must be odd. Defaults to 1023.

    Returns:
        y (torch.Tensor): Reverberated signal. Shape (bs, chs, seq_len).

    """
    assert num_bandpass_taps % 2 == 1, "num_bandpass_taps must be odd"

    bs, chs, seq_len = x.size()
    assert chs <= 2, "only mono/stereo signals are supported"

    # if mono copy to stereo
    if chs == 1:
        x = x.repeat(1, 2, 1)
        chs = 2

    # stack gains and decays into a single tensor
    band_gains = torch.stack(
        [
            band0_gain,
            band1_gain,
            band2_gain,
            band3_gain,
            band4_gain,
            band5_gain,
            band6_gain,
            band7_gain,
            band8_gain,
            band9_gain,
            band10_gain,
            band11_gain,
        ],
        dim=1,
    )
    band_gains = band_gains.unsqueeze(-1)

    band_decays = torch.stack(
        [
            band0_decay,
            band1_decay,
            band2_decay,
            band3_decay,
            band4_decay,
            band5_decay,
            band6_decay,
            band7_decay,
            band8_decay,
            band9_decay,
            band10_decay,
            band11_decay,
        ],
        dim=1,
    )
    band_decays = band_decays.unsqueeze(-1)

    # create the octave band filterbank filters
    filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, sample_rate)
    filters = filters.type_as(x)
    num_bands = filters.shape[0]

    # reshape gain, decay, and mix parameters
    band_gains = band_gains.view(bs, 1, num_bands, 1)
    band_decays = band_decays.view(bs, 1, num_bands, 1)
    mix = mix.view(bs, 1, 1)

    # generate white noise for IR generation
    pad_size = num_bandpass_taps - 1
    wn = torch.randn(bs * 2, num_bands, num_samples + pad_size).type_as(x)

    # filter white noise signals with each bandpass filter
    wn_filt = torch.nn.functional.conv1d(
        wn,
        filters,
        groups=num_bands,
        # padding=self.num_taps -1,
    )
    # shape: (bs * 2, num_bands, num_samples)
    wn_filt = wn_filt.view(bs, 2, num_bands, num_samples)

    # apply bandwise decay parameters (envelope)
    t = torch.linspace(0, 1, steps=num_samples).type_as(x)  # timesteps
    band_decays = (band_decays * 10.0) + 1.0
    env = torch.exp(-band_decays * t.view(1, 1, 1, -1))
    wn_filt *= env * band_gains

    # sum signals to create impulse shape: bs, 2, 1, num_samp
    w_filt_sum = wn_filt.mean(2, keepdim=True)

    # apply impulse response for each batch item (vectorized)
    x_pad = torch.nn.functional.pad(x, (num_samples - 1, 0))
    vconv1d = torch.vmap(partial(torch.nn.functional.conv1d, groups=2), in_dims=0)
    y = vconv1d(x_pad, torch.flip(w_filt_sum, dims=[-1]))

    # create a wet/dry mix
    y = (1 - mix) * x + mix * y

    return y


def simplified_reverb_optimized(
    x: torch.Tensor,
    sample_rate: int,
    room_size: torch.Tensor,
    damping: torch.Tensor,
    diffusion: torch.Tensor,
    pre_delay: torch.Tensor,
    wet_gain: torch.Tensor,
    ir_length: int = 65536,
    num_bands: int = 12,
) -> torch.Tensor:
    """
    A highly optimized reverb function using FFT-based convolution for maximum speed.
    All parameter tensors should have shape (bs, 1) or be broadcastable.
    """
    device = x.device
    bs, chs, seq_len = x.shape

    # --- 1. 상위 파라미터를 저수준 밴드 파라미터로 매핑 ---
    base_decay = (room_size.clamp(0.0, 1.0) * 4.0)
    # 0.0에서 1.0까지 변하는 기본 경사(ramp) 생성. 모양: (num_bands,)
    ramp = torch.linspace(0.0, 1.0, steps=num_bands, device=device)
    # damping 파라미터 클램핑. 모양: (bs, 1)
    damping_clamped = damping.clamp(0.0, 1.0)
    # 브로드캐스팅을 이용해 배치 단위의 감쇠 곡선 생성
    # (bs, 1) * (num_bands,) -> (bs, num_bands)
    damping_curve = 1.0 - damping_clamped * ramp
    band_decays = base_decay * damping_curve

    diffusion = diffusion.clamp(0.0, 1.0)
    flat_gains = torch.ones(bs, num_bands, device=device)
    random_gains = torch.rand(bs, num_bands, device=device) * 0.4 + 0.8
    band_gains = (1 - diffusion) * random_gains + diffusion * flat_gains
    band_gains = band_gains / torch.sqrt(torch.mean(band_gains**2, dim=-1, keepdim=True))

    # --- 2. 노이즈 기반 임펄스 응답(IR) 생성 (최적화 1 적용) ---
    num_bandpass_taps = 1023
    filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, sample_rate).type_as(x).squeeze(1)
    
    # FFT 연산을 위한 길이 계산
    n_fft = ir_length + num_bandpass_taps - 1

    # 노이즈와 필터를 FFT 도메인으로 변환
    noise = torch.randn(bs, num_bands, ir_length).type_as(x)
    fft_noise = torch.fft.rfft(noise, n=n_fft)
    fft_filters = torch.fft.rfft(filters, n=n_fft)

    # 주파수 도메인에서 필터링 (매우 빠름)
    filtered_noise_fft = fft_noise * fft_filters.unsqueeze(0)
    filtered_noise = torch.fft.irfft(filtered_noise_fft, n=n_fft)
    filtered_noise = filtered_noise[..., :ir_length] # 최종 길이로 자르기
    
    # 밴드별 Decay 및 Gain 적용
    t = torch.linspace(0, 1, steps=ir_length, device=device)
    decay_rate = (band_decays * 10.0) + 1.0
    envelope = torch.exp(-decay_rate.unsqueeze(-1) * t)
    shaped_noise = filtered_noise * envelope
    shaped_noise *= band_gains.unsqueeze(-1)

    # 모든 밴드를 합쳐서 최종 IR 생성
    impulse_response = torch.sum(shaped_noise, dim=1) # (B, ir_length)

    # --- 3. Pre-delay 적용 ---
    delay_samples = (pre_delay * sample_rate).long().squeeze(-1)
    
    padded_ir = torch.zeros_like(impulse_response)
    for i in range(bs):
        d = delay_samples[i].item()
        if d < 0 : d=0
        if d > 0 and d < ir_length:
            padded_ir[i, d:] = impulse_response[i, :-d]
        else:
            padded_ir[i, :] = impulse_response[i, :]

    # --- 4. 컨볼루션으로 리버브 적용 (최적화 2 적용) ---
    # fftconvolve은 채널별 처리를 위해 IR을 unsqueeze해야 함
    wet_signal = TAF.fftconvolve(x, padded_ir.unsqueeze(1), mode='same')
    
    # --- 5. 최종 Dry/Wet 믹스 ---
    wet_level = 0.7
    mix = wet_gain.clamp(0.0, 1.0).view(bs, 1, 1)
    
    return (1.0 - mix) * x + mix * (wet_level * wet_signal)

def professional_reverb_optimized(
    x: torch.Tensor,
    sample_rate: int,
    room_size: torch.Tensor,
    decay_time: torch.Tensor,
    damping: torch.Tensor,
    diffusion: torch.Tensor,
    pre_delay: torch.Tensor,
    wet_gain: torch.Tensor,
    ir_length: int = 65536,
    num_bands: int = 12,
) -> torch.Tensor:
    """
    Final, corrected version of the reverb. Diffusion and Decay are now correctly
    separated and applied.
    """
    device, bs, chs, seq_len = x.device, x.shape[0], x.shape[1], x.shape[2]
    def sigmoid(z, growth_rate=1.0, midpoint=5.0):
        return 1 / (1 + torch.exp(-growth_rate * (z - midpoint)))
    # --- 1. 감쇠(Decay) 및 댐핑(Damping) 파라미터로 'Envelope' 생성 ---
    rt60 = decay_time.clamp(0.1, ir_length/sample_rate)

    ramp = torch.linspace(0.0, 1.0, steps=num_bands, device=device)
    damping_curve = 1.0 - (damping.clamp(0.0, 1.0) * 0.7) * ramp
    band_rt60s = rt60 * damping_curve
    decay_constants = 6.91 / (band_rt60s * sample_rate + 1e-8)

    t_samples = torch.arange(ir_length, device=device)
    decay_constants_reshaped = decay_constants.unsqueeze(-1)
    
    # t_samples의 모양을 (ir_length,)로 유지
    # 브로드캐스팅: (bs, num_bands, 1) * (ir_length,) -> (bs, num_bands, ir_length)
    envelope = torch.exp(-decay_constants_reshaped * t_samples)
    
    # 최종적으로 shaped_noise와 곱해지기 위해 스테레오 채널 차원 추가
    # (bs, num_bands, ir_length) -> (bs, 1, num_bands, ir_length)
    envelope = envelope.unsqueeze(1)

    # --- 2. 노이즈 생성 및 감쇠 적용 ---
    num_bandpass_taps = 1023
    filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, sample_rate).type_as(x)
    n_fft_ir = ir_length + num_bandpass_taps - 1
    
    noise = torch.randn(bs, 2, num_bands, ir_length).type_as(x)
    
    fft_noise = torch.fft.rfft(noise, n=n_fft_ir, dim=-1)
    fft_filters = torch.fft.rfft(filters.squeeze(1), n=n_fft_ir, dim=-1)
    filtered_noise_fft = fft_noise * fft_filters.view(1, 1, num_bands, -1)
    filtered_noise = torch.fft.irfft(filtered_noise_fft, n=n_fft_ir, dim=-1)[..., :ir_length]
    
    # 노이즈에 Envelope를 먼저 적용하여 감쇠하는 잔향의 기초를 만듦
    shaped_noise = filtered_noise * envelope
    decaying_impulse = torch.sum(shaped_noise, dim=2)

    # --- 3. 디퓨전 적용 ---
    diffusion_g = diffusion.clamp(0.0, 1.0).squeeze(-1) * 0.5
    base_delay_lengths = [157, 239, 347, 479]
    n_fft_diffusion = ir_length

    H_total = torch.ones(bs, 2, n_fft_diffusion // 2 + 1, device=device, dtype=torch.complex64)
    for d_len_base in base_delay_lengths:
        # ⬇️ (핵심 수정) for 루프를 제거하고 벡터화 연산 시작
        
        # 1. 배치별 딜레이 길이 계산
        size_scale = 0.5 + (room_size.clamp(0.0, 10.0).squeeze(-1) / 10.0)
        d_len = (d_len_base * size_scale).long() # (bs,)
        max_d = torch.max(d_len) # 배치 내 최대 딜레이 길이
        
        # 2. max_d를 기준으로 계수 텐서 생성
        b_coeffs = torch.zeros(bs, max_d + 1, device=device)
        a_coeffs = torch.zeros(bs, max_d + 1, device=device)
        
        # 3. scatter_를 이용해 값 삽입
        # b_coeffs의 0번째 열에 diffusion_g 삽입
        b_coeffs[:, 0] = diffusion_g
        # b_coeffs의 각 행(i)의 d_len[i] 열에 1.0 삽입
        b_coeffs.scatter_(1, d_len.unsqueeze(-1), 1.0)
        
        # a_coeffs도 동일하게 처리
        a_coeffs[:, 0] = 1.0
        a_coeffs.scatter_(1, d_len.unsqueeze(-1), diffusion_g.unsqueeze(-1))
        
        # 4. 벡터화된 계수로 주파수 응답 한 번에 계산
        H_i = dasp_pytorch.signal.fft_freqz(b_coeffs, a_coeffs, n_fft=n_fft_diffusion)
        H_total *= H_i.unsqueeze(1)

    fft_ir = torch.fft.rfft(decaying_impulse, n=n_fft_diffusion, dim=-1)
    fft_diffused_ir = fft_ir * H_total
    diffused_ir = torch.fft.irfft(fft_diffused_ir, n=n_fft_diffusion, dim=-1)[..., :ir_length]
    peak_val = torch.max(torch.abs(diffused_ir), dim=-1, keepdim=True)[0]
    stable_ir = diffused_ir / (peak_val + 1e-8)
    stable_ir = torch.tanh(stable_ir)

    # --- 4. Pre-delay 적용 ---
    delay_samples = (pre_delay * sample_rate).clamp(0, ir_length / 2)
    delay_normalized = delay_samples * 2.0 / ir_length
    base_x_grid = torch.linspace(-1, 1, ir_length, device=device)
    shifted_x_grid = base_x_grid - delay_normalized
    shifted_y_grid = torch.zeros_like(shifted_x_grid)
    grid = torch.stack([shifted_x_grid, shifted_y_grid], dim=-1).unsqueeze(1)
    padded_ir = torch.nn.functional.grid_sample(stable_ir.unsqueeze(2), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(2)
    # --- 5. 최종 컨볼루션 및 믹스 ---
    if chs == 1: 
        x = x.repeat(1, 2, 1)
        
    full_conv_output = TAF.fftconvolve(x, padded_ir, mode='full')
    wet_signal = full_conv_output[..., :seq_len]
    
    mix = (wet_gain.clamp(0.0, 1.0)*0.4).view(bs, 1, 1)
    dry_amount = torch.cos(mix * torch.pi / 2)
    wet_amount = torch.sin(mix * torch.pi / 2)
    wet_level = 0.7
    
    return dry_amount * x + wet_amount * (wet_level * wet_signal)


def stereo_widener(x: torch.Tensor, sample_rate: float, width: torch.Tensor):
    """Stereo widener using mid-side processing.

    Args:
        x (torch.Tensor): Stereo audio tensor of shape (bs, 2, seq_len)
        sample_rate (float): Audio sample rate (Hz).
        width (torch.Tensor): Stereo width control. Higher is wider. has shape (bs)

    """
    bs, chs, seq_len = x.size()
    assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"

    sqrt2 = np.sqrt(2)
    mid = (x[..., 0, :] + x[..., 1, :]) / sqrt2
    side = (x[..., 0, :] - x[..., 1, :]) / sqrt2

    # amplify mid and side signal separately:
    mid *= 2 * (1 - width)
    side *= 2 * width

    # covert back to stereo
    left = (mid + side) / sqrt2
    right = (mid - side) / sqrt2

    return torch.stack((left, right), dim=-2)


def stereo_panner(x: torch.Tensor, sample_rate: float, pan: torch.Tensor):
    """Take a mono single and pan across the stereo field.

    Args:
        x (torch.Tensor): Monophonic audio tensor of shape (bs, num_tracks, seq_len).
        sample_rate (float): Audio sample rate (Hz).
        pan (torch.Tensor): Pan value on the range from 0 to 1.

    Returns:
        x (torch.Tensor): Stereo audio tensor with panning of shape (bs, num_tracks, 2, seq_len)
    """
    bs, num_tracks, seq_len = x.size()
    # first scale the linear [0, 1] to [0, pi/2]
    theta = pan * (np.pi / 2)

    # compute gain coefficients
    left_gain = torch.sqrt(((np.pi / 2) - theta) * (2 / np.pi) * torch.cos(theta))
    right_gain = torch.sqrt(theta * (2 / np.pi) * torch.sin(theta))

    # make stereo
    x = x.unsqueeze(1)
    x = x.repeat(1, 2, 1, 1)

    # apply panning
    left_gain = left_gain.view(bs, 1, num_tracks, 1)
    right_gain = right_gain.view(bs, 1, num_tracks, 1)
    gains = torch.cat((left_gain, right_gain), dim=1)
    x *= gains

    return x
