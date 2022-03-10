from numpy import array
from pylab import plot, legend, show

from fft_model import FFTmodel

data = array([1,2,3,2,1,0,1,2])
model = FFTmodel(number_of_harmonics=100)
forecast_data = model.predict(data)

plot(forecast_data, label = 'forecast')
plot(data, label = 'original data')
legend()
show()