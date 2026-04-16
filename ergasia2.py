import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 8192
L2 = 128  # max shiftings for autocorrelation

# Frequencies (Hz)
lambda1 = 0.12
lambda2 = 0.30
lambda4 = 0.19
lambda5 = 0.17
lambda3 = lambda1 + lambda2  # 0.42
lambda6 = lambda4 + lambda5  # 0.36

# Angular frequencies
omega1 = 2 * np.pi * lambda1
omega2 = 2 * np.pi * lambda2
omega3 = 2 * np.pi * lambda3
omega4 = 2 * np.pi * lambda4
omega5 = 2 * np.pi * lambda5
omega6 = 2 * np.pi * lambda6

# Random phases - independent, uniform on [0, 2*pi]
np.random.seed(42)  # for reproducibility
phi1 = np.random.uniform(0, 2 * np.pi)
phi2 = np.random.uniform(0, 2 * np.pi)
phi4 = np.random.uniform(0, 2 * np.pi)
phi5 = np.random.uniform(0, 2 * np.pi)
phi3 = phi1 + phi2
phi6 = phi4 + phi5

# --- Question 1: Construct X[k] ---
k = np.arange(N)

X = (np.cos(omega1 * k + phi1)
   + np.cos(omega2 * k + phi2)
   + np.cos(omega3 * k + phi3)
   + np.cos(omega4 * k + phi4)
   + np.cos(omega5 * k + phi5)
   + np.cos(omega6 * k + phi6))

# Plot X[k] (first 500 samples for visibility)
plt.figure(figsize=(12, 4))
plt.plot(k[:500], X[:500])
plt.xlabel('k')
plt.ylabel('X[k]')
plt.title('Question 1: Signal X[k] (first 500 samples)')
plt.grid(True)
plt.tight_layout()
plt.savefig('q1_signal.png', dpi=150)
plt.show()

# --- Question 2: Power Spectrum estimation via autocorrelation ---
# Estimate autocorrelation R_x[m] for m = -L2, ..., L2
# R_x[m] = (1/N) * sum_{k=0}^{N-1-|m|} X[k] * X[k+|m|]

lags = np.arange(-L2, L2 + 1)
Rx = np.zeros(len(lags))

for idx, m in enumerate(lags):
    m_abs = abs(m)
    Rx[idx] = (1.0 / N) * np.sum(X[:N - m_abs] * X[m_abs:N])

# Power spectrum: DFT of the windowed autocorrelation
# C_x(f) = sum_{m=-L2}^{L2} R_x[m] * exp(-j*2*pi*f*m)
# Evaluate on a dense frequency grid
Nf = 1024  # number of frequency points
f = np.linspace(0, 0.5, Nf)  # normalized frequency [0, 0.5]

Cx = np.zeros(Nf)
for i, fi in enumerate(f):
    Cx[i] = np.real(np.sum(Rx * np.exp(-1j * 2 * np.pi * fi * lags)))

# Plot power spectrum
plt.figure(figsize=(12, 5))
plt.plot(f, Cx)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$C_x^*(f)$')
plt.title(f'Question 2: Power Spectrum Estimate (L2 = {L2})')
plt.grid(True)

# Mark the expected frequencies
freqs_expected = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]
labels = [r'$\lambda_1$=0.12', r'$\lambda_2$=0.30', r'$\lambda_3$=0.42',
          r'$\lambda_4$=0.19', r'$\lambda_5$=0.17', r'$\lambda_6$=0.36']
for freq, label in zip(freqs_expected, labels):
    plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5, label=label)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('q2_power_spectrum.png', dpi=150)
plt.show()

print("Frequencies (Hz):", sorted(freqs_expected))
print("Done!")
