from sinusoid_generator import ConstantToneSinusoidGenerator
import numpy as np
import matplotlib.pyplot as plt

N = 8192
dt = 5e-9
TMAX = N * dt
TIME = np.linspace(0, TMAX, N, endpoint=False)

def testReal(s1, s2):
    return np.abs(np.dot(s1, s2))

def testComplex(s1, s2):
    return np.abs(np.dot(s1, np.conjugate(s2)))


def run_N_tests(N_tests, estim_param, sigma, signal_gen, signal_test, templates, params):
    total_error = 0
    mean = 0

    for run in range(N_tests):
        maxscore = 0
        best_params = (0,0)

        signal = signal_gen(aFreq=estim_param*2*np.pi, sigma=sigma, phase=np.random.uniform(0, 2*np.pi))

        for i in range(len(templates)):
            score = signal_test(signal, templates[i])
            
            if score > maxscore:
                maxscore = score
                best_params = params[i]

        #print(best_params[0], estim_param, "maxscore ", maxscore)
        squared_error = (best_params[0] - estim_param)**2
        total_error += squared_error
        mean += best_params[0]

    print("mean: ", mean/N_tests)
    MSE = total_error/N_tests
    return MSE


def fftTests(N_tests, estim_param, sigma, signal_gen, signal_test):
    total_error = 0
    freq = np.fft.fftfreq(8192*8) / dt
    df = freq[2] - freq[1]

    for run in range(N_tests):
        signal = np.pad(signal_gen(aFreq=estim_param*2*np.pi, sigma=sigma), (0, 8192*8 - N), 'constant')
        assert(len(signal) == 8192*8)

        sp = np.fft.fft(signal)
        V = np.absolute(sp)
        maxind = np.argmax(V)
        dk = - 0.5 * (V[maxind+1] - V[maxind-1]) / (V[maxind-1] - 2.*V[maxind] + V[maxind+1])
        best_param = np.abs(freq[maxind]+df*dk)
        squared_error = (best_param - estim_param)**2
        total_error += squared_error

    MSE = total_error/N_tests
    return MSE


fCyc = 52.123456e6
df_bin = 2400e6 / 8192
kBin = round(fCyc / df_bin)
fStart = fCyc - (0.01 * df_bin)
fEnd = fCyc + (0.01 * df_bin)
GENERATOR = ConstantToneSinusoidGenerator(time=TIME)

# real signals 

# templates, params = GENERATOR.getTemplatesReal(100, fStart=fStart, fEnd=fEnd)
x = []
y1 = []
for snr in np.logspace(-1, 4, 30, endpoint=False):
    s = np.sqrt(1./snr)
    # res = run_N_tests(10, fCyc, s, signal_gen=GENERATOR.signalReal, signal_test=testReal, templates=templates, params=params)
    res = fftTests(500, fCyc, s, GENERATOR.signalReal, testReal)
    print(s, res)
    x.append(snr)
    y1.append(res)

y1 = np.sqrt(np.array(y1))
plt.loglog(x, y1, color='r', label="real")


# complex signals 

# templates, params = GENERATOR.getTemplatesComplex(100, fStart=fStart, fEnd=fEnd)
x = []
y2 = []
for snr in np.logspace(-1, 4, 30, endpoint=False):
    s = np.sqrt(1./snr)
    # res = run_N_tests(10, fCyc, s, signal_gen=GENERATOR.signalComplex, signal_test=testComplex, templates=templates, params=params)
    res = fftTests(500, fCyc, s, GENERATOR.signalComplex, testComplex)
    print(s, res)
    x.append(snr)
    y2.append(res)

y2 = np.sqrt(np.array(y2))
plt.loglog(x, y2, color='b', label="complex")

# cramer-rao bound
x = []
crlb = []
print(N)
for snr in np.logspace(-1, 4, 100, endpoint=False):
    s = np.sqrt(1./snr)
    x.append(snr)
    crlb.append(3./ (2*np.pi**2*snr*dt**2.*N*(N**2. - 1.)))

crlb = np.sqrt(np.array(crlb))
plt.loglog(x, crlb, color='g', label="CRLB")

plt.legend(loc="lower left")
plt.xlabel("SNR")
plt.ylabel("RMSE")
plt.show()

