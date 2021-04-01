import pickle
import matplotlib.pyplot as plt
import numpy as np

lablenum = 10 # K
imagesize = 3072 # d
regularizationFactor = 0 # lamda

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
    X = dict[b'data'].transpose()
    y = dict[b'labels']
    Y = np.zeros((lablenum, X.shape[1]))
    for i in range(len(y)):
        Y[y[i], i] = 1
    return X, Y, y

def PreProcess(Data):
    return (Data - np.mean(Data, axis=0)) / np.std(Data, axis=0)

def GenWb(K=10, d=3072, m=0.0, di=0.01):
    return np.random.normal(m, di, (K, d)), \
           np.random.normal(m, di, (K, 1))

def EvaluateClassifier(X, W, b):
    s = np.matmul(W, X) + b
    return np.exp(s) / np.sum(np.exp(s), axis=0)

def ComputeCost(X, Y, W, b, lamda=regularizationFactor):
    lcross = np.diag(
        -np.log(np.matmul(Y.transpose(), EvaluateClassifier(X, W, b))))
    return (lcross.sum()) / X.shape[1] + lamda * np.sum(np.square(W))

def ComputeAccuracy(X, y, W, b):
    p = np.argmax(EvaluateClassifier(X, W, b), axis=0)
    err = np.count_nonzero(p - y) / X.shape[1]
    return (1 - err)

def ComputeGradients(X, Y, P, W, lamda=regularizationFactor):
    gBatch = -(Y - P)
    gW = np.matmul(gBatch, X.transpose()) / X.shape[1]
    gb = np.matmul(gBatch, np.ones((X.shape[1], 1))) / X.shape[1]
    return gW + 2*lamda*W, gb

def ComputeGradsNumSlow(X, Y, P, W, b, lamda=0, h=1e-6):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return grad_W, grad_b

def CheckGrads():
    trainX, trainHotY, trainY = LoadBatch('data_batch_1')
    w, b = GenWb()
    X = trainX[:, 0].reshape((imagesize, 1))
    Y = trainHotY[:, 0].reshape((lablenum, 1))
    y = trainY[:10]

    P = EvaluateClassifier(X, w, b)
    gdw, gdb = ComputeGradients(X, Y, P, w)
    gndw, gndb = ComputeGradsNumSlow(X, Y, P, w, b)

    sumw = np.absolute(gdw) + np.absolute(gndw)
    ret = np.average(np.absolute(gdw - gndw)) / np.amax(sumw)

    if ret < 1e-5:
        print('Check ComputeGradients passed!')
    else:
        print('Check ComputeGradients failed!')


if __name__ == '__main__':
    # # load the data
    # trainX, trainHotY, trainY = LoadBatch('data_batch_1')
    # valX, valHotY, valY = LoadBatch('data_batch_2')
    # testX, testHotY, testY = LoadBatch('test_batch')
    # # print(trainHotY)
    # # print(np.shape(trainHotY))
    # # print(trainY)
    # # print(np.shape(trainY))
    # # Montage(data[b'data'])

    # # pre-process the data
    # trainX = PreProcess(trainX)
    # valX = PreProcess(valX)
    # testX = PreProcess(testX)

    # # print(trainX)

    # # w, b = GenWb()
    # # p = EvaluateClassifier(trainX, w, b)
    # # print(p)
    # # print(np.shape(p))

    # # w, b = GenWb()
    # # c = ComputeCost(trainX, trainHotY, w, b, 0.01)
    # # print(c)

    # # w, b = GenWb()
    # # a = ComputeAccuracy(trainX, trainY, w, b)
    # # print(a)

    CheckGrads()

