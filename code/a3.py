import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from Layer import NeuralNetwork


def read_and_store(path="data/seg_train/seg_train"):
    t_set = list()
    label_counter = 0  # 0->buildings
    # 1-> forest and so on
    for train_folder in os.listdir(path):
        folder = path + "\\" + train_folder
        for filename in os.listdir(folder):
            img = cv2.imread(folder + "\\" + filename, 0).astype(np.uint8)  # gray img
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalization
            out = img.flatten()  # to obtain 1d image array
            t_set.append([out, label_counter])  # out is 900,1 vector and train_folder is label
        label_counter += 1
    return t_set


def read_test():
    test = list()
    folder = "data/seg_test"
    for filename in os.listdir(folder):
        img = cv2.imread(folder + "\\" + filename, 0).astype(np.uint8)  # gray img
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalization
        out = img.flatten()  # to obtain 1d image array
        test.append([out])
    return test


def plotting(x, y, y2, y3, label1="Train", label2="Loss",
             xlabel="Epoch", ylabel="Accuracy", title="title"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.plot(x, y, label=label1)
    ax1.plot(x,y2,label="Validation")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()

    ax2.plot(x, y3, label=label2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(label2)
    ax2.legend()
    plt.show()


def process(train_set, validation_set):
    act_function = [[NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_derivation, "sigmoid"],
                    [NeuralNetwork.tanh, NeuralNetwork.der_tanh, "tanh"]
                    ]
    hid_lay_list = [[],[100],[10,6]]
    lr_list = [0.001,0.01,0.05,0.1]
    batch_list = [2,4,128]
    counter = 1

    for hidden_size in hid_lay_list:
        for act, der_act, a in act_function:
            for lr in lr_list:
                for b_s in batch_list:
                    NN = NeuralNetwork(hidden_layers=hidden_size, activation_func=act,
                                       der_func=der_act)
                    np.random.shuffle(train_set)
                    epoch_acc = NN.train2(train_set, validation_set, 100, lr, batch_size=b_s)
                    pickle.dump(NN, open(("model" + str(counter)), "wb"))
                    counter += 1

                    # NN=pickle.load(open("model"+str(counter),"rb"))

                    e = 0
                    c = 0
                    for i in range(len(validation_set)):
                        o = NN.forward(validation_set[i][0])
                        error = validation_set[i][1] - np.argmax(o)
                        if error == 0:
                            c += 1
                    print("Accuracy of validation is {:.2f}%".format(c * 100 / len(validation_set)))
                    string = "Model {} Hidden layer {}, Activation function {}, Epoch is 200, Learning Rate is {}, Batch size is {} ".format(
                        counter - 1, hidden_size, a, lr, b_s)
                    print(string)
                    print("------------------------------------------------------------------------------")
                    print()
                    epo = np.array([i[0] for i in epoch_acc])
                    acc = np.array([i[1] for i in epoch_acc])
                    val_acc = np.array([i[2] for i in epoch_acc])
                    lo_ss = np.array([i[3] for i in epoch_acc])
                    plotting(epo, acc, val_acc, lo_ss, title=string)



train_set = read_and_store(path="data/seg_train/seg_train")
test_set = read_test()
validation_set = read_and_store(path="data/seg_dev/seg_dev")

pickle.dump(train_set,open("train_set", 'wb'))
pickle.dump(test_set,open("test_set", 'wb'))
pickle.dump(validation_set,open("validation", 'wb'))


"""with open("train_set", "rb") as f:
    train_set = pickle.load(f)
with open("test_set", "rb") as f:
    test_set = pickle.load(f)

with open("validation", "rb") as f:
    validation_set = pickle.load(f)"""

process(train_set, validation_set)
