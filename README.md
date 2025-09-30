# multiple_coherence_analyse
multiple coherence analyse for vibration and noise correlation analysis

### Uasge example
```python
from multiple_coherence_analyse import multiple_coherence_analyse
import numpy as np
import matplotlib.pyplot as plt

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
```