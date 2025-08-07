from signal_interface import SignalInterface

import numpy as np

class ConstantToneSinusoidGenerator(SignalInterface):
    def __init__(self, time: np.ndarray) -> None:
        self.time = time
        self.N = len(self.time)

    def Phi(self, aFreq: float, phase: float):
        return aFreq * self.time + phase

    def templateReal(self, amplitude: float = 1, aFreq: float = 0.25, phase: float = 0) -> np.ndarray:
        return np.real(self.templateComplex(amplitude, aFreq, phase))

    def templateComplex(self, amplitude: float = 1, aFreq: float = 0.25, phase: float = 0) -> np.ndarray:
        return amplitude * (np.cos(self.Phi(aFreq, phase)) + 1j * np.sin(self.Phi(aFreq, phase)))

    def signalReal(self, amplitude: float = 1, aFreq: float = 0.25, phase: float = 0, sigma: float = 1) -> np.ndarray:
        return np.real(self.signalComplex(amplitude, aFreq, phase, sigma))

    def signalComplex(self, amplitude: float = 1, aFreq: float = 0.25, phase: float = 0, sigma: float = 1) -> np.ndarray:
        return self.templateComplex(amplitude, aFreq, phase) + self.noiseComplex(self.N, sigma)

    def getTemplatesReal(self, N, fStart, fEnd):
        templates = []
        params = []

        for template_freq in np.linspace(fStart, fEnd, N, endpoint=False):
            for phase in np.linspace(0, 2.*np.pi, 200):
                params.append((template_freq, 0))
                templates.append(self.templateReal(aFreq=template_freq*2*np.pi, phase=phase))

        return (templates, params)

    def getTemplatesComplex(self, N, fStart, fEnd):
        templates = []
        params = []

        for template_freq in np.linspace(fStart, fEnd, N, endpoint=False):
            for phase in np.linspace(0, 2.*np.pi, 200):
                params.append((template_freq, 0))
                templates.append(self.templateComplex(aFreq=template_freq*2*np.pi, phase=phase))

        return (templates, params)
