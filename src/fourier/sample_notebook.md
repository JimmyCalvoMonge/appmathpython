```markdown
# Fourier Analysis

## Introduction

Fourier Analysis is a powerful mathematical tool used to examine functions, signals, or data sets in order to analyze their frequency components. The transformation techniques involved allow us to convert data from the time domain to the frequency domain and vice versa. Its applications are widespread across various fields, such as physics, engineering, and signal processing. The fundamental processes in Fourier analysis include Fourier Series, Fourier Transform, Discrete Fourier Transform, and Fast Fourier Transform.

## Fourier Series

### Theory

A Fourier Series is a way to represent a periodic function as a sum of simple sine and cosine waves. A periodic function \( f(x) \) with period \( 2\pi \) can be expressed as:

\[ f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos(nx) + b_n \sin(nx) \right) \]

where the coefficients are given by:

\[ a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx \]

\[ a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx \]

\[ b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx \]

### Examples

Let's see how we can calculate the Fourier Series coefficients of a simple function using Python:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the periodic function
def f(x):
    return np.pi - x if np.pi <= x <= 2 * np.pi else x

# Compute Fourier coefficients
def fourier_coefficients(n_max, func, T):
    a_0 = (1 / T) * quad(func, 0, T)[0]
    a_n = [(2 / T) * quad(lambda x: func(x) * np.cos(n * (2 * np.pi / T) * x), 0, T)[0] for n in range(1, n_max + 1)]
    b_n = [(2 / T) * quad(lambda x: func(x) * np.sin(n * (2 * np.pi / T) * x), 0, T)[0] for n in range(1, n_max + 1)]
    return a_0, a_n, b_n

# Calculate coefficients
a_0, a_n, b_n = fourier_coefficients(10, f, 2 * np.pi)

print("a_0 =", a_0)
print("a_n =", a_n)
print("b_n =", b_n)
```

The above code calculates the Fourier coefficients for a simple sawtooth wave defined over one period. Note the use of `scipy.integrate.quad` for precise integration.

## Fourier Transform

### Theory

The Fourier Transform generalizes the concept of Fourier Series to non-periodic functions. It transforms a time-domain signal into its constituent frequencies. The Fourier Transform \( \hat{f}(\omega) \) of a function \( f(t) \) is given by:

\[ \hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i \omega t} \, dt \]

The Inverse Fourier Transform is expressed as:

\[ f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega) e^{i \omega t} \, d\omega \]

### Examples

Let's implement the Fourier Transform using Python and evaluate it for a Gaussian function:

```python
from scipy.fft import fft, fftfreq
import numpy as np

# Define a Gaussian function
def gaussian(t):
    return np.exp(-t ** 2)

# Sampling parameters
N = 1024  # Number of sample points
L = 10.0  # Length of the interval
t = np.linspace(-L/2, L/2, N)

# Fourier Transform using FFT
gaussian_ft = fft(gaussian(t))
freqs = fftfreq(N, L / N)

# Plot
plt.plot(freqs, np.abs(gaussian_ft))
plt.title("Fourier Transform of Gaussian Function")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
```

## Discrete Fourier Transform (DFT)

### Theory

The Discrete Fourier Transform (DFT) is the Fourier Transform for discrete, finite data sets. For a sequence \( x_0, x_1, ..., x_{N-1} \), the DFT is defined by:

\[ X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi kn/N} \]

where \( k = 0, 1, ..., N-1 \). It transforms discrete signals from the time domain to the frequency domain.

### Examples

Below is an example of calculating the DFT using Python. Here, the NumPy library will be used to perform the transformation:

```python
import numpy as np

# Sample data: simple sinusoidal wave
N = 8
n = np.arange(N)
x = np.sin(2 * np.pi * n / N)
X = np.fft.fft(x)

# Display results
print("Signal: ", x)
print("DFT: ", X)

# Visualize the result
plt.stem(n, np.abs(X), use_line_collection=True)
plt.title("Discrete Fourier Transform")
plt.xlabel("Frequency component")
plt.ylabel("Magnitude")
plt.show()
```

## Fast Fourier Transform (FFT)

### Theory

The Fast Fourier Transform (FFT) is an efficient algorithm to compute the DFT and its inverse. It reduces the complexity from \( O(N^2) \) to \( O(N \log N) \) making it feasible to work with large datasets.

### Examples

We will use NumPy's efficient FFT implementation to compute FFT of a real-world audio signal:

```python
import numpy as np
from scipy.io import wavfile

# Load a WAV file
sample_rate, data = wavfile.read('example.wav')

# Compute FFT
fft_data = np.fft.fft(data)
frequencies = np.fft.fftfreq(len(data), 1/sample_rate)

# Plot
plt.plot(frequencies, np.abs(fft_data))
plt.title("FFT of the audio signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 10**4)  # Limit the x-axis to 10 kHz
plt.show()
```

## Applications in Signal Processing

### Theory

Fourier Analysis is heavily utilized in signal processing. It allows us to isolate frequency components from signals, filter noise, and compress data. 

### Examples

In practice, Fourier Transform can remove unwanted noise from audio signals by filtering. Let's demonstrate a basic audio denoising using Fourier Transform:

```python
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read audio file
sample_rate, data = wavfile.read('noisy_audio.wav')

# Apply Fourier Transform
fourier_data = np.fft.fft(data)

# Filter - remove frequencies higher than a threshold
threshold = 1000  # Hz
fourier_data[np.abs(np.fft.fftfreq(len(fourier_data), 1 / sample_rate)) > threshold] = 0

# Inverse Fourier Transform
denoised_data = np.fft.ifft(fourier_data)

# Save and play cleaned audio
wavfile.write('denoised_audio.wav', sample_rate, denoised_data.astype(np.int16))

# Visualization
plt.subplot(211)
plt.title("Original Signal Spectrum")
plt.plot(np.abs(np.fft.fft(data)))
plt.subplot(212)
plt.title("Filtered Signal Spectrum")
plt.plot(np.abs(fourier_data))
plt.show()
```

The code filters out high-frequency noise from an audio file. By adjusting the threshold, we can preserve desired audible frequencies while eliminating others.

In conclusion, Fourier Analysis is a foundational technique with wide applications in science and engineering, allowing deep insights and operations within the frequency domain.
```