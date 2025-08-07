from vaunix_generator import VaunixGenerator
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from signal_interface import SignalInterface

N = 8192
dt = 2**14/(4.8e9) * 2
TMAX = N * dt
TIME = np.linspace(0, TMAX, N, endpoint=False)


def run_N_tests(N_tests, generator, period0, templates, params, sigma):
    total_error = 0
    mean = 0

    for _ in range(N_tests):
        best_score = 0
        convolution = []
        period_est = 0

        phase = np.random.uniform()
        period = period0
        signal = generator.signalReal(169e-6, period, phase*period, sigma)

        for i in range(len(templates)):
            conv = sp.signal.correlate(signal, templates[i], method="fft")

            maxscore = np.max(np.abs(conv))
            
            if maxscore > best_score:
                best_score = maxscore
                period_est = params[i]
                convolution = conv

        convtimes = np.linspace(-(N-1) * dt, (N-1)*dt, 2*N-1)
        a = convolution[np.argmax(convolution)-1]
        b = convolution[np.argmax(convolution)]
        c = convolution[np.argmax(convolution)+1]
        deltak = 0.5 * (a-c) / (a-2*b+c)

        final_estimate = (convtimes[np.argmax(convolution)] + period_est)%period_est + deltak * dt
        final_estimate = (final_estimate + period_est)%period_est

        error = (period_est - final_estimate) - phase*period
        squared_error = error**2
        total_error += squared_error
        mean += error

    # print("bias: ", mean/N_tests/dt, "bins")
    MSE = total_error/N_tests
    return MSE

def complexTests(N_tests, generator, period0, phi0, templates, params, sigma):
    total_error = 0
    mean = 0

    for _ in range(N_tests):
        best_score = 0
        convolution = []
        period_est = 0

        period = period0
        t0 = np.random.uniform(0, period)
        period_est = period
        phi = phi0


        signal = generator.signalComplex(169e-6, period, t0, phi, sigma)

        for i in range(len(templates)):
            conv = sp.signal.correlate(signal, templates[i], method="fft")
            conv = np.abs(conv)

            maxscore = np.max(conv)
            
            if maxscore > best_score:
                best_score = maxscore
                convolution = conv

        convtimes = np.linspace(-(N-1) * dt, (N-1)*dt, 2*N-1)
        a = convolution[np.argmax(convolution)-1]
        b = convolution[np.argmax(convolution)]
        c = convolution[np.argmax(convolution)+1]
        deltak = 0.5 * (a-c) / (a-2*b+c)

        final_estimate = (convtimes[np.argmax(convolution)] + period_est)%period_est + deltak * dt
        final_estimate = (final_estimate + period_est)%period_est

        error = (period_est - final_estimate) - t0
        squared_error = error**2
        total_error += squared_error
        mean += error

    # print("bias: ", mean/N_tests/dt, "bins")
    MSE = total_error/N_tests
    return MSE

period0 = 0.021003
phi0 = 0.123456789

GENERATOR = VaunixGenerator(time=TIME)

# real signals 
templates, params = GENERATOR.getTemplatesReal(period0)
x1 = []
y1 = []

for snr in np.logspace(-1, 4, 20, endpoint=False):
    sigma = np.sqrt(1./snr)
    res = run_N_tests(60, GENERATOR, period0, templates, params, sigma)
    print("RESULTS: sigma:", sigma, "RMSE:", np.sqrt(res)/dt, "bins")
    x1.append(snr)
    y1.append(res)

x1 = np.array(x1)
y1 = np.sqrt(np.array(y1))/dt

#complex signals
templates, params = GENERATOR.getTemplatesComplex(period0, 1)
x2 = []
y2 = []

for snr in np.logspace(-1, 4, 10, endpoint=False):
    sigma = np.sqrt(1./snr)
    res = complexTests(60, GENERATOR, period0, phi0, templates, params, sigma)
    print("RESULTS: sigma:", sigma, "RMSE:", np.sqrt(res)/dt, "bins")
    x2.append(snr)
    y2.append(res)

x2 = np.array(x2)
y2 = np.sqrt(np.array(y2))/dt

CRLB = 1/x1 * 5.97985375e-16*36
CRLB = np.sqrt(CRLB)/dt

plt.loglog(x1, y1, color='r', label="real")
plt.loglog(x2, y2, color='b', label="complex")
plt.loglog(x1, CRLB, color='g', label='CRLB')

plt.legend(loc="lower left")
plt.xlabel("SNR")
plt.ylabel("RMSE")
plt.show()

