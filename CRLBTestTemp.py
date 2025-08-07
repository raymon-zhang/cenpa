import matplotlib.pyplot as plt
import numpy as np
from scipy import special
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 19})

fig = plt.figure(figsize=[13,7])
ax = fig.gca()

# Fixing random state for reproducibility
np.random.seed(198388904)

R = 1.
kB=1.38e-23
T = np.logspace(-10,2,25)
dt = 5e-9
eps = 0.5
Pe = 1.1e-15
Pe*=eps
t = np.arange(0.0, 1e-3, dt)
NFFT = 8192*8  # the length of the windowing segments
Fs = int(1.0 / dt)  # the sampling frequency
B = Fs / 2.
# create a transient "chirp"
tStart=0
#tLength = 1e-6
tLength = 8192 * dt
fCyc = 52.123456e6
A = np.sqrt(Pe*R)
sigma2 = kB*T*R*B

mask = np.where(np.logical_and(t > tStart, t < (tStart+tLength)), 1.0, 0.0)
s1 = np.sqrt(Pe*R)* (np.cos(2. * np.pi * fCyc * t) + 0j*np.sin(2.*np.pi*fCyc*t)) * mask

peaks = []
t2 = t[0:NFFT-1]
freq = np.fft.fftfreq(t2.shape[-1]) / ( dt)
df = freq[2] - freq[1]


stdPeaks =0*T
stdCRLB =0*T

nTrials=25

for j in range(len(T)):
    print(j)
    for i in range(nTrials):
        nse = np.sqrt(sigma2[j]/2.)*np.random.normal(size=len(t)) + 0j*np.sqrt(sigma2[j]/2.)*np.random.normal(size=len(t)) 
        x = (s1 + nse)*mask  # the signal
        x = x[0:NFFT-1]
        sp = np.fft.fft(x)
        V = np.absolute(sp)
        V2 = np.absolute(sp)**2
        maxind = np.argmax(V)
        dk = - 0.5 * (V[maxind+1] - V[maxind-1]) / (V[maxind-1] - 2.*V[maxind] + V[maxind+1])
        peaks = np.append(peaks,np.abs(freq[maxind]+df*dk))

    peaks-=fCyc
    print(peaks)
    stdPeaks[j] = np.sqrt(np.mean(np.square(peaks)))
    peaks = []

###########Compute CRLB##################
nSamples = tLength/dt
stdCRLB = np.sqrt(3*sigma2 / (2.*np.pi**2. * Pe*R*dt**2.*nSamples*(nSamples**2. - 1.)))
###########################################

stdUP = np.sqrt(12)/(4.*np.pi *tLength) * np.ones(len(sigma2))
SNR = A**2 / sigma2

plt.loglog(SNR,stdUP**2,'--k',label="Gabor's Limit")
plt.loglog(SNR,stdCRLB**2,'-k',label="CRLB")
plt.loglog(SNR, stdPeaks**2,'.k',label="Interpolated")

plt.legend()
plt.xlabel('SNR')
plt.ylabel(r'$\sigma_f^2$ ($\mathrm{Hz}^2$)')
plt.subplots_adjust(bottom=0.14, right=0.93, left=0.10, top=0.93)
plt.show()
