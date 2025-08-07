import numpy as np

class SignalInterface:
    def __init__(self) -> None:
        """Constructor"""
        print("Signal Instantiated")
    
    def noiseReal(self, N: int, sigma: float) -> np.ndarray:
        """Generate real-valued AWGN"""
        return np.real(self.noiseComplex(N, sigma))
    
    def noiseComplex(self, N: int, sigma: float) -> np.ndarray:
        """Generate complex-valued AWGN"""
        return np.sqrt(sigma**2/2.)*np.random.normal(size=N) + 1j*np.sqrt(sigma**2/2.)*np.random.normal(size=N) 
        # return np.random.normal(0, sigma, size=N) + 0j*np.random.normal(0, sigma, size=N) 

    @staticmethod
    def normalize(signal):
        return signal / np.sqrt(np.sum(np.square(np.abs(signal))))
