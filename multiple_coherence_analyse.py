from typing import Optional, List
import numpy as np

def delay_estimate(tgt_signal: np.ndarray, ref_signal: np.ndarray, max_delay_points: Optional[int]=None):
    """Estimate the delay between two signals using cross-correlation."""
    """
    Estimate the delay between two signals using cross-correlation.
    Parameters
    ----------
    tgt_signal : np.ndarray
        Target signal, [D].
    ref_signal : np.ndarray
        Reference signal, [D].
    max_delay_points : Optional[int], optional
        Maximum delay points to consider, by default None.
    Returns
    -------
    int
        Estimated delay in points.
    """
    assert len(tgt_signal) == len(ref_signal), "Signals must have the same length."
    S1 = np.fft.rfft(tgt_signal)
    S2 = np.fft.rfft(ref_signal)
    C = S1 * S2.conj()
    # C /= np.abs(C) + 1e-12 # gcc_phat operate, not need when process long time signal
    cc = np.fft.irfft(C)
    import scipy
    cc = scipy.signal.correlate(tgt_signal, ref_signal, mode='same')
    max_idx = np.argmax(cc)
    if max_idx >= len(cc) // 2:
        max_idx -= len(cc)
    if max_delay_points is not None:
        max_idx = max(min(max_idx, max_delay_points), -max_delay_points)
    return max_idx

def multiple_coherence_analyse(signals: np.ndarray, noise_channels: List[int]=[1, 2], vibra_channels: List[int]=[3, 4, 5],
                               sr: int=3000, n_fft: int=1024, max_delay_ms: int=10, low_freq: int=50, high_freq: int=500):
    """Analyse multiple coherence between noise and vibration signals.
    
    Parameters
    ----------
    signals : np.ndarray
        Input signals, [N, D].
    noise_channels : List[int], optional
        Indices of noise channels, by default [1, 2].
    vibra_channels : List[int], optional
        Indices of vibration channels, by default [3, 4, 5].
    sr : int, optional
        Sampling frequency, by default 3000.
    n_fft : int, optional
        Frame length for STFT, by default 1024.
    max_delay_ms : int, optional
        Maximum delay in milliseconds for delay estimation, by default 10.
    low_freq : int, optional
        Low frequency bound for coherence calculation, by default 50Hz.
    high_freq : int, optional
        High frequency bound for coherence calculation, by default 500Hz.

    Returns
    -------
    f_range : np.ndarray
        Frequency range for coherence, [F].
    Cnv_range : np.ndarray
        Coherence values in the frequency range, [F].
    Cnv_avg : float
        Average coherence in the specified frequency range.
    Dnv : np.ndarray
        Estimated delays between noise and vibration channels, [len(vibra_channels)].
    """
    from scipy.signal import csd
    N, D = signals.shape
    assert N > max(noise_channels) and N > max(vibra_channels), "Channel index out of range."

    n_nums, v_nums = len(noise_channels), len(vibra_channels)
    max_delay_points = int(max_delay_ms / 1000 * sr) if max_delay_ms is not None else None
    # 计算噪声信号的自功率谱
    f, Pnn_list = None, []
    for ch in noise_channels:
        f, Pnn = csd(signals[ch], signals[ch], fs=sr, nperseg=n_fft)
        Pnn_list.append(Pnn)
    
    f_len = len(f)
    Cnv = np.zeros(f_len) # 噪声振动相干性数组
    Dnv = np.zeros(v_nums) # 噪声振动因果性数组
    # 计算振动信号的互功率谱
    Pvv = np.zeros((v_nums, v_nums, f_len), dtype=np.complex128)
    for i, ch_i in enumerate(vibra_channels):
        for j, ch_j in enumerate(vibra_channels):
            # Pvv[i, j, :] = csd(signals[ch_i], signals[ch_j], fs=sr, nperseg=n_fft)
            if i <= j:
                _, Pvv_ij = csd(signals[ch_i], signals[ch_j], fs=sr, nperseg=n_fft)
                Pvv[i, j, :] = Pvv_ij
                if i != j:
                    Pvv[j, i, :] = np.conj(Pvv_ij)
    
    # 增广功率谱矩阵
    Pnnv = np.zeros((v_nums+1, v_nums+1, f_len), dtype=np.complex128)
    Pnnv[1: v_nums+1, 1: v_nums+1, :] = Pvv

    for i in range(n_nums):
        Pnn = Pnn_list[i]
        Pnnv[0, 0, :] = Pnn
        for j in range(v_nums):
            _, Pnv = csd(signals[noise_channels[i]], signals[vibra_channels[j]], fs=sr, nperseg=n_fft)
            Pnnv[0, j+1, :] = Pnv
            Pnnv[j+1, 0, :] = np.conj(Pnv)
        # 计算多重相干性
        for f_idx in range(f_len):
            Pnnv_f = Pnnv[:, :, f_idx]
            Pnn_f = Pnn[f_idx]
            Pvv_f = Pvv[:, :, f_idx]
            Cnv[f_idx] += np.abs(1 - np.linalg.det(Pnnv_f) / (Pnn_f * np.linalg.det(Pvv_f)))
        # 计算因果性
        for j in range(v_nums):
            Dnv[j] += delay_estimate(signals[noise_channels[i]], signals[vibra_channels[j]], max_delay_points=max_delay_points)

    Dnv /= n_nums
    Cnv /= n_nums
    low_freq = low_freq if low_freq else 0
    high_freq = high_freq if high_freq else f[f_len-1]
    Cnv_range = Cnv[(f >= low_freq) & (f <= high_freq)]
    f_range = f[(f >= low_freq) & (f <= high_freq)]
    Cnv_avg = np.mean(Cnv_range)

    return f_range, Cnv_range, Cnv_avg, Dnv

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 生成测试信号
    sr = 3000
    t = np.arange(0, 10, 1/sr)
    noise1 = np.random.randn(len(t)) * 0.5
    noise2 = np.random.randn(len(t)) * 0.5
    vibra1 = np.sin(2 * np.pi * 100 * t) + np.random.randn(len(t)) * 0.1 + np.roll(noise1, 5)
    vibra2 = np.sin(2 * np.pi * 200 * t) + np.random.randn(len(t)) * 0.1 + np.roll(noise2, -3)
    vibra3 = np.sin(2 * np.pi * 300 * t) + np.random.randn(len(t)) * 0.1 + np.roll(noise1, 2) + np.roll(noise2, -4)

    signals = np.vstack([noise1, noise2, vibra1, vibra2, vibra3])

    f_range, Cnv_range, Cnv_avg, Dnv = multiple_coherence_analyse(signals, noise_channels=[0, 1], vibra_channels=[2, 3, 4],
                                                                  sr=sr, n_fft=1024, max_delay_ms=10, low_freq=50, high_freq=500)

    print("Average Coherence:", Cnv_avg)
    print("Delays (in samples):", Dnv)

    plt.figure()
    plt.plot(f_range, Cnv_range)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.title("Multiple Coherence between Noise and Vibration Signals")
    plt.grid()
    plt.show()
