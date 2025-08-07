from signal_interface import SignalInterface

import numpy as np

class VaunixGenerator(SignalInterface):
    def __init__(self, time: np.ndarray) -> None:
        self.time = time
        self.N = len(self.time)
        # self.vaunix_freq = 17.98817814e6
        self.vaunix_freq = 0

    def templateReal(self, Ton, period, t0) -> np.ndarray:
        phase = t0 / period
        duty = Ton / period

        res = [0.] * self.time.shape[0]

        interval = self.time[1]-self.time[0]

        for i in range(len(self.time)):
            start = np.mod(self.time[i] + phase*period, period)
            end = start+interval

            if end > period:
                res[i] = ((end-period) / (end-start))**2 * 255
            elif end > duty*period and start < duty*period:
                res[i] = ((duty*period - start) / (end-start))**2 * 255
            elif start >= duty*period:
                res[i] = 0
            else:
                res[i] = 255

        return np.array(res)/np.sqrt(np.sum(np.square(res))) * 1e4

    def signalReal(self, Ton, period, t0, noiseCoefficient=1) -> np.ndarray:
        phase = t0 / period
        duty = Ton / period

        res = [0.] * self.time.shape[0]

        off = 0
        on = 98.3
        noise = np.random.randn(*self.time.shape)

        interval_length = self.time[1]-self.time[0]

        for i in range(len(self.time)):
            start = np.mod(self.time[i] + phase*period, period)
            end = start+interval_length

            noise = np.random.normal()

            if end > period:
                res[i] = ((end-period) / (end-start))**2 * on
            elif end > duty*period and start < duty*period:
                res[i] = ((duty*period - start) / (end-start))**2 * on
            elif start >= duty*period:
                res[i] = off
            else:
                res[i] = on

            res[i] += noise*res[i]*noiseCoefficient

        return np.array(res).astype(np.uint8)

    def templateComplex(self, Ton, period, t0, phi0) -> np.ndarray:
        phase = t0 / period
        duty = Ton / period

        res = [0.] * self.time.shape[0]

        interval_length = self.time[1]-self.time[0]

        for i in range(len(self.time)):
            start = np.mod(self.time[i] + phase*period, period)
            end = start+interval_length

            complex_phase = (phi0 + self.vaunix_freq*2.*np.pi*self.time[i])

            if end > period:
                res[i] = ((end-period) / (end-start))**2 * 255
            elif end > duty*period and start < duty*period:
                res[i] = ((duty*period - start) / (end-start))**2 * 255
            elif start >= duty*period:
                res[i] = 0
            else:
                res[i] = 255

            res[i] *= np.exp(1j*complex_phase)

        return np.array(res)/np.sqrt(np.sum(np.abs(res)))


    def signalComplex(self, Ton, period, t0, phi0, noiseCoefficient=1) -> np.ndarray:
        phase = t0 / period
        duty = Ton / period

        res = [0.] * self.time.shape[0]

        off = 0
        on = 98.3

        interval_length = self.time[1]-self.time[0]

        for i in range(len(self.time)):
            start = np.mod(self.time[i] + phase*period, period)
            end = start+interval_length

            complex_phase = (phi0 + self.vaunix_freq*2.*np.pi*self.time[i])

            noise = np.sqrt(1/2.)*(np.random.normal() + 1j*np.random.normal())

            if end > period:
                res[i] = ((end-period) / (end-start))**2 * on
            elif end > duty*period and start < duty*period:
                res[i] = ((duty*period - start) / (end-start))**2 * on
            elif start >= duty*period:
                res[i] = off
            else:
                res[i] = on

            res[i] *= np.exp(1j*complex_phase)

            res[i] += noise*np.abs(res[i])*noiseCoefficient

        return np.array(res)/np.sqrt(np.sum(np.abs(res)))

    def getTemplatesReal(self, period):
        templates = []
        params = []

        params.append((period))
        templates.append(self.templateReal(169e-6, period, 0))

        return (templates, params)

    def getTemplatesComplex(self, period, Nphi):
        templates = []
        params = []

        for phi0 in np.linspace(0.123456789, 0.123456789, Nphi):
            params.append((phi0))
            templates.append(self.templateComplex(169e-6, period, 0, phi0))

        return (templates, params)
