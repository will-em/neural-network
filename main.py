import numpy as np
import pickle
from matplotlib import pyplot as plt

#Sigmoid gradient:
def sigmoidGradient(z):

    return sigmoid(z)*(1-sigmoid(z))


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoidGradient(z)

currentPath = "/Users/William/github/neural-network"

with open(currentPath + "/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_imgs = data[1]
test_imgs = data[0]
train_labels = data[3]
test_labels = data[2]
train_labels_one_hot = data[5]
test_labels_one_hot = data[4]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size



input_layer_size = image_pixels
hidden_layer_size = 10
output_layer_size = no_of_different_labels
sizes = [input_layer_size, hidden_layer_size, output_layer_size]

#Choose Cost Function, "CrossEntropyCost" or "QuadraticCost":
cost = CrossEntropyCost



#Sigmoid activation function:
def sigmoid(z):
    return 1 / (1+np.exp(-1*z))

#ReLU activation function:
def ReLU(z):
    return np.maximum(0, z)

#SoftMax:
def softMax(z):
    return np.exp(z) / np.sum(np.exp(z))

#Weight and bias initialization
def initialization(sizes):

    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x)/np.sqrt(x)
                    for x, y in zip(sizes[:-1], sizes[1:])]

    return weights, biases

def feedForward(weights, biases, input):

    a = input
    for w, b in zip(weights, biases):
        a = sigmoid(np.dot(w, a)+ b)

    return a

def backPropagation(weights, biases,  x, y):

    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b = [np.zeros(b.shape) for b in biases]

    activation = x
    activation = np.reshape(activation, (activation.shape[0], 1))
    activations = [x]
    zs = []
    for w, b in zip(weights, biases):
        z = np.dot(w, activation) + b
        zs.append(z)

        activation = sigmoid(z)
        activations.append(activation)

    #Backprop:

    delta = cost.delta(zs[-1], activations[-1], y)
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    nabla_b[-1] = delta

    delta = np.dot(weights[-1].transpose(), delta) * sigmoidGradient(zs[-2])
    nabla_w[-2] = np.dot(delta, activations[-3].transpose())
    nabla_b[-2] = delta


    return nabla_w, nabla_b

weights = []
biases = []

def accuracy(weights, biases, num, n):
    num_of_correct = 0
    for i in range(n*num, (n+1)*num):
        input = np.reshape(test_imgs[i, :], (test_imgs[i, :].shape[0], 1))
        a = feedForward(weights, biases, input)
        correct = np.reshape(test_labels_one_hot[i, :], (test_labels_one_hot[i, :].shape[0], 1))

        if np.argmax(a)-np.argmax(correct)==0:
            num_of_correct+=1

    print("Accuracy: " + str(100*num_of_correct/num) + "%")

weights, biases = initialization(sizes)

m = train_imgs.shape[0]

mini_batch_size = 100
learning_rate = 0.25
lamb = 0
num_of_epoch = 1


for currentEpoch in range(0, num_of_epoch):
    print("Epoch: " + str(currentEpoch+1))
    n = 0
    while (n+1)*mini_batch_size <= m:
        nabla_w = [np.zeros(w.shape) for w in weights]
        nabla_b = [np.zeros(b.shape) for b in biases]
        currentCost = 0
        for i in range(n*mini_batch_size, (n+1)*mini_batch_size):

            x = np.reshape(train_imgs[i, :], (image_pixels, 1))
            test = train_labels_one_hot[i, :]
            y = np.reshape(test, (test.shape[0], 1))

            currentCost += cost.fn(feedForward(weights, biases, x), y)
            (nabla_w_delta, nabla_b_delta) = backPropagation(weights, biases, x, y)
            nabla_w = [nab + delt for nab, delt in zip(nabla_w, nabla_w_delta)]
            nabla_b = [nab + delt for nab, delt in zip(nabla_b, nabla_b_delta)]

        weights = [w - learning_rate/mini_batch_size*nw for w, nw in zip(weights, nabla_w)]
        biases = [b - learning_rate/mini_batch_size*nb for b, nb in zip(biases, nabla_b)]

        currentCost = currentCost/mini_batch_size
        print("Cost: " + str(currentCost))
        n += 1

    accuracy(weights, biases, 100, currentEpoch)

plt.ion()
test_imgs = test_imgs[1000:, :]
np.random.shuffle(test_imgs)

for i in range(0, test_imgs.shape[0]-1):
    image = np.reshape(test_imgs[i, :], (test_imgs[i, :].shape[0], 1))
    imageShow = test_imgs[i, :].reshape((image_size, image_size))

    a = feedForward(weights, biases, image)
    print("Guess: ", np.argmax(a),"\n")
    plt.imshow(imageShow, interpolation='nearest')
    plt.show()
    inStr = input("Enter to proceed, 0 to exit: ")
    print("")
    if inStr is "0":
        break
    else:
        plt.close()
