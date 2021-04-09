import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import math

lable_num = 10 # K
image_size = 3072 # d
regularization_factor = 0 # lamda

# mini batch parameters
minibatch_lambda = 0
GDparameters = namedtuple('GDparameters', ['n_batch', 'eta', 'n_epochs'])
GDps = GDparameters(100, 0.001, 40)

def DrawGraphs(cost_train_list, cost_val_list, loss_train_list, loss_val_list, acc_train_list, acc_val_list):
    fig = plt.figure(figsize=(30, 10))
    fig.add_subplot(131)
    plt.plot(cost_train_list, color='green', label='Training Cost')
    plt.plot(cost_val_list, color='red', label='Validation Cost')
    plt.legend(loc='upper right')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.title('Cost function output')

    fig.add_subplot(132)
    plt.plot(loss_train_list, color='green', label='Training Accuracy')
    plt.plot(loss_val_list, color='red', label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss function output')

    fig.add_subplot(133)
    plt.plot(acc_train_list, color='green', label='Training Accuracy')
    plt.plot(acc_val_list, color='red', label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy function output')

    plt.savefig('result.png')


def Montage(W, save='w.png'):
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
    Y = np.zeros((lable_num, X.shape[1]))
    for i in range(len(y)):
        Y[y[i], i] = 1
    return X, Y, y

def GetMeanStd(Data):
    return np.mean(Data), np.std(Data)

def PreProcess(x, mean, std):
    return (x - mean) / std

def GenWb(K=10, d=3072, m=50, mean=0.0, di=0.01):
    return np.random.normal(mean, 1/math.sqrt(d), (m, d)), \
           np.random.normal(mean, 1/math.sqrt(m), (K, m)), \
           np.random.normal(mean, 1/math.sqrt(d), (m, 1)), \
           np.random.normal(mean, 1/math.sqrt(m), (K, 1))

def EvaluateClassifier(X, W1, b1, W2, b2):
    s1 = np.matmul(W1, X) + b1
    h = ActivationFun(s1)
    s = np.matmul(W2, h) + b2
    return np.exp(s) / np.sum(np.exp(s), axis=0), h

def ActivationFun(s):
    """ ReLU """
    return np.maximum(0, s)

def ComputeCost(X, Y, W1, b1, W2, b2, lamda=regularization_factor):
    p, _h = EvaluateClassifier(X, W1, b1, W2, b2)
    lcross = np.diag(-np.log(np.matmul(Y.transpose(), p)))
    loss = lcross.sum()
    return (loss / X.shape[1]) + \
           lamda * (np.sum(np.square(W1)) + np.sum(np.square(W2))), loss / X.shape[1]

def ComputeAccuracy(X, y, W1, b1, W2, b2):
    p, _h = EvaluateClassifier(X, W1, b1, W2, b2)
    p = np.argmax(p, axis=0)
    err = np.count_nonzero(p - y) / X.shape[1]
    return (1 - err)

def ComputeGradients(X, Y, P, h, W1, W2, lamda=regularization_factor):
    gBatch = -(Y - P)
    gW2 = np.matmul(gBatch, h.transpose()) / X.shape[1]
    gb2 = np.matmul(gBatch, np.ones((X.shape[1], 1))) / X.shape[1]

    gBatch = np.matmul(W2.transpose(), gBatch)
    gBatch = gBatch * (h > 0)
    gW1 = np.matmul(gBatch, X.transpose()) / X.shape[1]
    gb1 = np.matmul(gBatch, np.ones((X.shape[1], 1))) / X.shape[1]

    return gW1 + 2*lamda*W1, gb1, gW2 + 2*lamda*W2, gb2

def ComputeGradsNumSlow(X, Y, P, W1, b1, W2, b2, lamda=0, h=1e-5):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros((W1.shape[0], 1))
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros((W2.shape[0], 1))

    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= h
        c1, _h = ComputeCost(X, Y, W1, b1_try, W2, b2, lamda)

        b1_try = np.array(b1)
        b1_try[i] += h
        c2, _h = ComputeCost(X, Y, W1, b1_try, W2, b2, lamda)

        grad_b1[i] = (c2-c1) / (2*h)

    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1, _h = ComputeCost(X, Y, W1, b1, W2, b2_try, lamda)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2, _h = ComputeCost(X, Y, W1, b1, W2, b2_try, lamda)

        grad_b2[i] = (c2-c1) / (2*h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i,j] -= h
            c1, _h = ComputeCost(X, Y, W1_try, b1, W2, b2, lamda)

            W1_try = np.array(W1)
            W1_try[i,j] += h
            c2, _h = ComputeCost(X, Y, W1_try, b1, W2, b2, lamda)

            grad_W1[i,j] = (c2-c1) / (2*h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i,j] -= h
            c1, _h = ComputeCost(X, Y, W1, b1, W2_try, b2, lamda)

            W2_try = np.array(W2)
            W2_try[i,j] += h
            c2, _h = ComputeCost(X, Y, W1, b1, W2_try, b2, lamda)

            grad_W2[i,j] = (c2-c1) / (2*h)

    return grad_W1, grad_b1, grad_W2, grad_b2

def CheckGrads():
    trainX, trainHotY, trainY = LoadBatch('data_batch_1')
    w1, w2, b1, b2 = GenWb()
    X = trainX[:, 0].reshape((image_size, 1))
    Y = trainHotY[:, 0].reshape((lable_num, 1))
    y = trainY[:10]

    P, h = EvaluateClassifier(X, w1, b1, w2, b2)
    gdw1, gdb1, gdw2, gdb2= ComputeGradients(X, Y, P, h, w1, w2)
    gndw1, gndb1, gndw2, gndb2 = ComputeGradsNumSlow(X, Y, P, w1, b1, w2, b2)

    sumw1 = np.absolute(gdw1) + np.absolute(gndw1)
    ret1 = np.average(np.absolute(gdw1 - gndw1)) / np.amax(sumw1)

    sumw2 = np.absolute(gdw2) + np.absolute(gndw2)
    ret2 = np.average(np.absolute(gdw2 - gndw2)) / np.amax(sumw2)

    if ret1 < 1e-5 and ret2 < 1e-5:
        print('Check ComputeGradients passed!')
    else:
        print('Check ComputeGradients failed!')

def GetEta(t, eta_min=1e-5, eta_max=1e-1, ns=500):
    interval = 2 * ns
    while t > interval:
        t -= interval

    if t <= 2*ns:
        return eta_min + (t/ns)*(eta_max-eta_min)
    else:
        return eta_max - ((t-ns)/ns)*(eta_max-eta_min)

def GetMinibatches(X, Y, GDparams=GDps, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    minibatches_num = math.floor(m / GDparams.n_batch)
    for k in range(minibatches_num):
        minibatch_X = shuffled_X[:, k * GDparams.n_batch:(k+1) * GDparams.n_batch]
        minibatch_Y = shuffled_Y[:, k * GDparams.n_batch:(k+1) * GDparams.n_batch]
        mini_batches.append((minibatch_X, minibatch_Y))
    if m % GDparams.n_batch != 0:
        minibatch_X = shuffled_X[:, minibatches_num * GDparams.n_batch:]
        minibatch_Y = shuffled_Y[:, minibatches_num * GDparams.n_batch:]
        mini_batches.append((minibatch_X, minibatch_Y))

    return mini_batches

def MiniBatchGD(X, Y, y, ValX, ValY, Valy, W1, b1, W2, b2, GDparams=GDps, MinibatchLambda=minibatch_lambda):
    cost_train_list = []
    cost_val_list = []
    loss_train_list = []
    loss_val_list = []
    acc_train_list = []
    acc_val_list = []

    t = 1
    cycles = 2
    n_s = 2*math.floor(45000/100)
    for i in range(GDparams.n_epochs):
        if t > 2*cycles*n_s:
            break
        # seed += 1
        minibatches = GetMinibatches(X, Y, GDparams)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            P, h = EvaluateClassifier(minibatch_X, W1, b1, W2, b2)
            gd_w1, gd_b1, gd_w2, gd_b2 = ComputeGradients(minibatch_X, minibatch_Y, P, h, W1, W2, MinibatchLambda)
            eta = GetEta(t, ns=n_s)
            W1 -= eta * gd_w1
            b1 -= eta * gd_b1
            W2 -= eta * gd_w2
            b2 -= eta * gd_b2
            t += 1

        P_train, _h = EvaluateClassifier(X, W1, b1, W2, b2)
        P_val, _h = EvaluateClassifier(ValX, W1, b1, W2, b2)
        cost_train, h_train = ComputeCost(X, Y, W1, b1, W2, b2, MinibatchLambda)
        cost_val, h_val = ComputeCost(ValX, ValY, W1, b1, W2, b2, MinibatchLambda)
        acc_train = ComputeAccuracy(X, y, W1, b1, W2, b2)
        acc_val = ComputeAccuracy(ValX, Valy, W1, b1, W2, b2)

        cost_train_list.append(cost_train)
        cost_val_list.append(cost_val)
        loss_train_list.append(h_train)
        loss_val_list.append(h_val)
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Train cost is: ' + str(cost_train))
        print('Train accuracy is: ' + str(acc_train))
        print('******************************************')
        print('Validation cost is: ' + str(cost_val))
        print('Validation accuracy is: ' + str(acc_val))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    DrawGraphs(cost_train_list, cost_val_list, loss_train_list, loss_val_list, acc_train_list, acc_val_list)
    # Montage(W)
    return W1, b1, W2, b2


if __name__ == '__main__':
    print('Check gradient computation accurary...')
    CheckGrads()
    print('Training start, loading the data...')
    # load the data
    x1, Y1, y1 = LoadBatch('data_batch_1')
    x2, Y2, y2 = LoadBatch('data_batch_2')
    x3, Y3, y3 = LoadBatch('data_batch_3')
    x4, Y4, y4 = LoadBatch('data_batch_4')
    x5, Y5, y5 = LoadBatch('data_batch_5')

    trainX = np.concatenate((x1, x2, x3, x4, x5), axis=1)
    trainHotY = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)
    trainY = y1 + y2 + y3 + y4 + y5

    valX = trainX[:, trainX.shape[1]-5000:]
    valHotY = trainHotY[:, trainHotY.shape[1]-5000:]
    valY = trainY[-5000:]

    trainX = trainX[:, :trainX.shape[1]-5000]
    trainHotY = trainHotY[:, :trainHotY.shape[1]-5000]
    trainY = trainY[:-5000]

    testX, testHotY, testY = LoadBatch('test_batch')

    # pre-process the data
    print('Preprocess the data...')
    meanX, stdX = GetMeanStd(trainX)
    trainX = PreProcess(trainX, meanX, stdX)
    valX = PreProcess(valX, meanX, stdX)
    testX = PreProcess(testX, meanX, stdX)

    print('Start training...')
    W1, W2, b1, b2 = GenWb()
    W1, b1, W2, b2 = MiniBatchGD(trainX, trainHotY, trainY, valX, valHotY, valY, W1, b1, W2, b2)
    print('Final test cost is: ' + str(ComputeCost(testX, testHotY, W1, b1, W2, b2)))
    print('Final test accuracy is: ' + str(ComputeAccuracy(testX, testY, W1, b1, W2, b2)))

