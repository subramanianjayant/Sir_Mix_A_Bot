import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import wavfile

FILE_DIR = '/Users/subra/Desktop/git/Sir_Mix_A_Bot/wav_files/'
filename = 'OblivionPiazzola.wav'

#returns ndarray formatted sound pressure/tone data
def get_array(filename):
    sample_rate,data = wavfile.read(filename)
    data = data/(2.**15)
    return data,sample_rate

#main method prints statistics about data
if __name__=='__main__':
    data, sample_rate = get_array(FILE_DIR+filename)
    print('sample_rate = %s' % sample_rate)
    print(data.shape)
