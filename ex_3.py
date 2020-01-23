import random
import sys

import numpy as np

hidden_layers_size = 150
inputPixel = 784
numOfClasses = 10
epochs = 30
learning_rate = 0.01

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
relU = lambda x: np.maximum(0, x)
relU_derivative = lambda x: 1.0 if x > 0 else 0.0


def relu_deriative(x):
    return (x > 0) * 1


def relu(x):
    return np.maximum(0, x)


def back_pro(forward, y):
    # get x, z1, h1, z2, h2 from the parameters dictionary
    x, z1, h1, z2, h2, w2 = [forward[element] for element in ('x', 'z1', 'h1', 'z2', 'h2', 'w2')]

    # creates new vector for the predict label in a size of output_size x 1
    vec_y = np.zeros((10, 1))
    vec_y[int(y)] = 1
    dz2_dl = (h2 - vec_y)  # dL/dz2
    dW2 = np.dot(dz2_dl, h1.T)  # dL/dz2 * dz2/dw2

    db2 = dz2_dl  # dL/dz2 * dz2/db2 =  dL/dz2
    dz1 = np.dot(forward['w2'].T, (h2 - vec_y)) * relu_deriative(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1

    dW1 = np.dot(dz1, x.T.reshape(1, 784))  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1

    return {'db1': db1, 'dw1': dW1, 'db2': db2, 'dw2': dW2}


def softmax(z2):
    e_x = np.exp(z2 - np.max(z2))
    return e_x / e_x.sum(axis=0)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def feed_forward(x, y, params):
    w1, b1, w2, b2 = [params[element] for element in ('w1', 'b1', 'w2', 'b2')]
    x = np.array(x)
    x.shape = (inputPixel, 1)
    np.transpose(b1)
    np.transpose(b2)
    z1 = np.add(np.dot(w1, x), b1)
    h1 = relu(z1)
    z2 = np.add(np.dot(w2, h1), b2)
    h2 = softmax(z2)
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for element in params:
        ret[element] = params[element]

    return ret


def validation(params, validation_x, validation_y):
    success_rate = 0
    loss = 0
    for x, y in zip(validation_x, validation_y):
        forward = feed_forward(x, y, params)
        arg = forward['h2']
        y_hat = np.argmax(arg)
        if y_hat == int(y):
            success_rate += 1
    accuracy = success_rate / float(np.shape(validation_x)[0])
    avg = loss / np.shape(validation_x)[0]
    return avg, accuracy


def updateParams(param, gradients, learning):
    w1, b1, w2, b2 = [param[key] for key in ('w1', 'b1', 'w2', 'b2')]
    dw1, dw2, db1, db2 = [gradients[key] for key in ('dw1', 'dw2', 'db1', 'db2')]
    w1 = w1 - learning * dw1
    w2 = w2 - learning * dw2
    b1 = b1 - learning * db1
    b2 = b2 - learning * db2
    return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}


def makeShuffle(train_x, labels):
    zip_train_labels = list(zip(train_x, labels))
    random.shuffle(zip_train_labels)
    new_train_x, new_train_y = zip(*zip_train_labels)
    return new_train_x, new_train_y


def train(train_x, train_y, param, echo, learning, validation_x, validation_y):
    shapeSize = train_x.shape[0]
    for i in range(echo):
        print(i)
        sum_loss = 0
        train_x, train_y = makeShuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            forward = feed_forward(x, y, param)
            gradients = back_pro(forward, y)
            param = updateParams(param, gradients, learning)
        #val_loss, accurate = validation(param, validation_x, validation_y)
        #print(i, sum_loss / shapeSize, val_loss, accurate * 100)
    return param


def predict(params, x):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x_reshped = x.reshape(784, 1)
    z1 = np.dot(w1, x_reshped) + b1  # w1*x + b1
    h1 = relu_deriative(z1)
    z2 = np.dot(w2, h1) + b2
    return softmax(z2)


def testIt(test_x, params):
    file_test_y = open("test_y", "w")
    for x in test_x:
        forward_ret = feed_forward(x, 1, params)
        str_class = str(np.argmax(forward_ret['h2']))
        file_test_y.write(str_class + "\n")
    file_test_y.close()


def loadFilesAndNormalized():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    train_x = np.divide(train_x, 255)
    test_x = np.divide(test_x, 255)
    return train_x, train_y, test_x


def splitData(train_x, train_y):
    sizeOfData = int(inputPixel * 0.2)
    train_x = train_x[sizeOfData:]
    train_y = train_y[sizeOfData:]
    validation_x = train_x[:sizeOfData]
    validation_y = train_y[:sizeOfData]
    return train_x, train_y, validation_x, validation_y


def main():
    train_x, train_y, test_x = loadFilesAndNormalized()
    W1 = np.random.uniform(-0.08, 0.08, (hidden_layers_size, inputPixel))
    W2 = np.random.uniform(-0.08, 0.08, (numOfClasses, hidden_layers_size))
    b1 = np.random.uniform(-0.08, 0.08, (hidden_layers_size, 1))
    b2 = np.random.uniform(-0.08, 0.08, (numOfClasses, 1))
    params = {'w1': W1, 'b1': b1, 'w2': W2, 'b2': b2}
    #train_x, train_y, validation_x, validation_y = splitData(train_x, train_y)
    validation_x = 0
    validation_y = 0
    weights = train(train_x, train_y, params, epochs, learning_rate, validation_x, validation_y)
    testIt(test_x, weights)


if __name__ == '__main__':
    main()
