from typing import Tuple 

from numpy import polyfit, angle, cos, pi, zeros, ndarray, array
from numpy.fft import fft, fftfreq

class FFTmodel:
    """
    use fast fourier transform (FFT)
    to model any function
    and make predictions
    """
    def __init__(self, number_of_harmonics:int=5) -> None:
        self.number_of_harmonics = 1+ 2*number_of_harmonics 

    def predict(self, data:ndarray, forecast_size:int=10) -> ndarray:
        frequencies = fftfreq(data.size)
        data_adjusted, unadjustment = self.adjust(data, forecast_size)    
        coefficients=fft(data_adjusted)
        indexes = sorted(
            range(data.size),
            key=lambda index:abs(coefficients[index]), 
            reverse=True
        )
        reconstructed_data_adjusted = sum(map(
            lambda index: self.construct_wave(
                coefficient=coefficients[index],
                frequency=frequencies[index],
                size=data.size,
                forecast_size=forecast_size
            ),
            indexes[:self.number_of_harmonics],
        ))
        return reconstructed_data_adjusted + unadjustment

    @staticmethod
    def adjust(data:ndarray, forecast_size:int) -> Tuple[ndarray,ndarray]:
        magnitudes = range(data.size)
        initial_coefficient,_ = polyfit(magnitudes, data, 1) 
        return (
            data - initial_coefficient * magnitudes, 
            initial_coefficient * range(data.size + forecast_size)
        )

    @staticmethod
    def construct_wave(coefficient:complex,size:int,forecast_size:int,frequency:float) -> ndarray:
        amplitude = FFTmodel.magnitude(coefficient,size)
        phase = FFTmodel.direction(coefficient) 
        return amplitude * cos(2 * pi * frequency * range(size+forecast_size) + phase)

    @staticmethod
    def magnitude(coefficient:complex, size:int) -> float:
        return abs(coefficient) / size  
    
    @staticmethod
    def direction(coefficient:complex) -> float:
        return angle(coefficient)
