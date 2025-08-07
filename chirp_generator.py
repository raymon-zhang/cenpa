from signal_interface import SignalInterface

import numpy as np

class ChirpGenerator(SignalInterface):
    def __init__(self, time: np.ndarray) -> None:
        self.time = time
        self.N = len(self.time)

    def Phi(self, omega0, alpha0, phase):
        return omega0 * self.time + alpha0/2*(self.time**2) + phase

    def templateReal(self, omega0, alpha0, amplitude = 1, phase = 1) -> np.ndarray:
        return np.real(self.templateComplex(omega0, alpha0, amplitude, phase))

    def templateComplex(self, omega0, alpha0, amplitude = 1, phase = 1) -> np.ndarray:
        return amplitude * (np.cos(self.Phi(omega0, alpha0, phase)) + 1j * np.sin(self.Phi(omega0, alpha0, phase)))

    def signalReal(self, omega0, alpha0, amplitude = 1, phase = 1, sigma = 1) -> np.ndarray:
        return np.real(self.signalComplex(omega0, alpha0, amplitude, phase, sigma))

    def signalComplex(self, omega0, alpha0, amplitude = 1, phase = 1, sigma = 1) -> np.ndarray:
        return self.templateComplex(omega0, alpha0, amplitude, phase) + self.noiseComplex(self.N, sigma)

    def getTemplatesReal(self, Nf, Nalpha, fStart, fEnd, alphaStart, alphaEnd):
        templates = []
        params = []

        for template_freq in np.linspace(fStart, fEnd, Nf):
            for template_alpha in np.linspace(alphaStart, alphaEnd, Nalpha):
                params.append((template_freq, template_alpha))
                templates.append(self.templateReal(omega0=template_freq*2*np.pi, alpha0 = template_alpha))

        return (templates, params)

    def getTemplatesComplex(self, Nf, Nalpha, fStart, fEnd, alphaStart, alphaEnd):
        templates = []
        params = []

        for template_freq in np.linspace(fStart, fEnd, Nf):
            for template_alpha in np.linspace(alphaStart, alphaEnd, Nalpha):
                params.append((template_freq, template_alpha))
                templates.append(self.templateComplex(omega0=template_freq*2*np.pi, alpha0 = template_alpha))

        return (templates, params)
