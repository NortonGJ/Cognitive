import SoapySDR, warnings
import matplotlib.pyplot as plt
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
from SoapySDR import * #SOAPY_SDR_ constants
from tqdm import trange
import numpy as np, scipy as sp
from .func import mergeData
def conv(value, From1=-120, From2=-30, To1=-1, To2=0):
    return (value - From1) / (From2 - From1) * (To2 - To1) + To1

datadir = './SDRPlay-RPS1/data_50'

warnings.filterwarnings("ignore")
SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
SoapySDR.registerLogHandler(None)

# Data transfer settings
rx_chan = 0             # RX1 = 0, RX2 = 1
N = 64512               # Number of complex samples per transfer # Block size [64512]
fs = 10e6               # Radio sample Rate
freq = 15e6             # LO tuning frequency in Hz
fMEG = freq//1e6
fsMEG = fs//1e6
use_agc = False         # Use or don't use the AGC  # True or False
timeout_us = int(5e6)
rx_bits = 16            # The AIR-T's ADC is 16 bits

# Initialize the SDRplay receiver using SoapySDR
sdr = SoapySDR.Device(dict(driver = "sdrplay"))    # Create SDRplay instance
sdr.setSampleRate(SOAPY_SDR_RX, rx_chan, fs)       # Set sample rate
sdr.setFrequency(SOAPY_SDR_RX, rx_chan, freq)      # Tune the LO
sdr.setGainMode(SOAPY_SDR_RX, rx_chan, use_agc)    # Set the gain mode
gain = 40
sdr.setGain(SoapySDR.SOAPY_SDR_RX, rx_chan, "TUNER", gain)

# Receive Signal
    
# Create data buffer and start streaming samples to it
stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])  # Setup data stream
block_size_0 = sdr.getStreamMTU(stream)
# print("Block size_0:     ", block_size_0)
# print("Block size_Stream:", N)
buffer_size = 2 * N  # I+Q
# print("Buffer size:      ", buffer_size)
buffer_format = np.int16

iStart = 0
iEnd = 200

Tall = 300; La = 50
dT = 0.1
NdT = int(dT*fs)
NdTpass = int(dT * fs * 0.8)
StreamData = np.empty((Tall, La))
sdr.activateStream(stream) #start streaming  # this turns the radio on
for index in trange(iStart, iEnd, 1):
    for ii in trange(Tall):
        data = np.empty(N * 16)
        
        for i in range(16):
            buffer = np.empty(buffer_size, buffer_format) # Create memory buffer for data stream
            
            # Read the samples from the data buffer
            sr = sdr.readStream(stream, [buffer], N, timeoutUs = timeout_us)
            rc = sr.ret   # number of samples read or the error code
            
            # Convert interleaved shorts (received signal) to numpy.complex64 normalized between [-1, 1]
            s0 = buffer.astype(float) / np.power(2.0, rx_bits-1)
            s = (s0[::2] + 1j*s0[1::2])
    
            #Накидываем сюда дату 16 раз (64512 * 16 = 10 032 192)
            #(1 млн отсчетов = 0.1 сек, т.к. частота дискретизации 10 МГц)
            data[i * 64512: (i+1) * 64512] = s
    
        firstSecData = data[0:NdT]
        spectr = sp.fft.fftshift(sp.fft.fft(firstSecData, 1*10**6)/(1*10**6))
        new_spectr = 20*sp.log10(np.abs(spectr))
        new_spectr = new_spectr[NdT//10:9*NdT//10]
        new_spectr = new_spectr.reshape((La, NdTpass//La))
        new_spectr_vec =  np.max(new_spectr, axis = 1)
        StreamData[ii] = new_spectr_vec
    with open (f'{datadir}/{index}.npy', 'wb') as File:
        np.save(File, StreamData)
sdr.deactivateStream(stream)
sdr.closeStream(stream)

mergeData(datadir, '15_50_60000_noise', iStart, iEnd)