#http://cs229.stanford.edu/proj2011/FavaroLewisSchlesinger-IcaForMusicalSignalSeparation.pdf
#http://www.cs.tut.fi/sgn/arg/music/tuomasv/unsupervised_virtanen.pdf
#https://github.com/MTG/DeepConvSep/blob/master/transform.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import time
import sklearn

FILE_DIR = '/Users/subra/Desktop/git/Sir_Mix_A_Bot/wav_files/'
filename = 'OblivionPiazzola.wav'

#reads audio data into ndarray
sample_rate,data = wavfile.read(FILE_DIR+filename)
n_channels = 2
sample_width = 2
n_frames = len(data)

#plot original signals
plt.figure('Original Stereo')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(np.arange(n_frames)/sample_rate,data[:,0],alpha = 0.2,color = 'blue')
plt.plot(np.arange(n_frames)/sample_rate,data[:,1],alpha = 0.2,color = 'red')

#ICA decomposition
from sklearn.decomposition import FastICA
ica = FastICA()
print('Training ICA...')
start_time = time.time()
ica.fit(data)
time = time.time()-start_time
print('Training complete after %f sec' % time)
transformed_data = ica.transform(data)
A = ica.mixing_
assert np.allclose(data,np.dot(transformed_data,A.T)+ica.mean_)

#Plots transformed data
plt.figure('Separated Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(np.arange(n_frames)/sample_rate,transformed_data[:,0],alpha = 0.2,color = 'blue')
plt.plot(np.arange(n_frames)/sample_rate,transformed_data[:,1],alpha = 0.2,color = 'red')
plt.show()

#saves transformed data channels as separate files
for x in range(0,2):
    transformed_data_slice = transformed_data[:,x]
    transformed_data_slice = transformed_data_slice.tobytes()
    outfile = wave.open(FILE_DIR+'Oblivion'+str(x)+'.wav','w')
    outfile.setnchannels(n_channels)
    outfile.setsampwidth(sample_width)
    outfile.setframerate(sample_rate)
    outfile.setnframes(n_frames)
    outfile.writeframes(transformed_data_slice)
    outfile.close()
