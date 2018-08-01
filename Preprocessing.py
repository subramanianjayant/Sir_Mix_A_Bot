import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import sklearn

FILE_DIR = '/Users/subra/Desktop/git/Sir_Mix_A_Bot/wav_files/'
filename = 'OblivionPiazzola.wav'

infile = wave.open(FILE_DIR+filename,'r')
n_channels,sample_width,sample_rate,n_frames,comptype,compname = infile.getparams()
data = infile.readframes(n_frames)
infile.close()

from numpy.fft import fft
data = np.fromstring(data,'Int16').reshape(-1,2)
data = data[:,0]

#from sklearn.decomposition import FastICA
#ica = FastICA()
#data = ica.fit_transform(data)
#plt.plot(data[:,0],alpha = 0.2,color = 'blue')
plt.plot(data,alpha = 0.2,color = 'red')
plt.show()

data = data.tobytes()
outfile = wave.open(FILE_DIR+'OblivionWritten.wav','w')
outfile.setnchannels(n_channels)
outfile.setsampwidth(sample_width)
outfile.setframerate(sample_rate)
outfile.setnframes(n_frames)
outfile.writeframes(data)
outfile.close()
