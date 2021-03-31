import pickle
import matplotlib.pyplot as plt
import numpy as np

def Montage(W, save='mygraph.png'):
	""" Display the image for each label in W """
	_fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.savefig(save)

def LoadBatch(filename):
    """ Read data from file """
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def PreProcess(Data):
    Data = np.transpose(Data)
    return np.transpose((Data - np.mean(Data, axis=0)) / np.std(Data, axis=0))

def GenWb(K=10, d=3072, m=0.0, di=0.01):
    return np.random.normal(m, di, (K, d)), \
           np.random.normal(m, di, (K, 1))

if __name__ == '__main__':
    # load the data
    trainingData = LoadBatch('data_batch_1')
    validationData = LoadBatch('data_batch_2')
    testData = LoadBatch('test_batch')
    # Montage(data[b'data'])

    # pre-process the data
    trainingX = PreProcess(trainingData[b'data'])
    validationX = PreProcess(validationData[b'data'])
    testX = PreProcess(testData[b'data'])
