import tensorflow.compat.v1 as tf
import numpy as np
from cifar10_train import *
import matplotlib.pyplot as plt

def compute_acc(predict, vali):
    acc = 0
    for i in range(len(vali)):
        if np.argmax(predict[i])==vali[i]:
            acc+=1
    return acc/len(vali)
    
label_class = ['airplane  ',
               'automobile',
               'bird      ',
               'cat       ',
               'deer      ',
               'dog       ',
               'frog      ',
               'horse     ',
               'ship      ',
               'truck     ']
    
def render_eval(vali_data,vali_labels,predict_labels):
    num_pics=min(len(vali_data),10)
    print('----------------------------')
    print('Validation images:')
    plt.figure(figsize=(10,5))
    for i in range(num_pics):
        plt.subplot(1,num_pics,i+1)
        plt.imshow(vali_data[i])
    plt.show()
    print('----------------------------')
    print('Labels:')
    for i in range(num_pics):
        print('picture %d: \t ground truth: %s \t predict: %s'%(i,
                                                               label_class[int(vali_labels[i])],
                                                               label_class[np.argmax(predict_labels[i])]))
    print('----------------------------')
    
def get_image():
    #vali_data = cv2.imread('vali_data/img_0.png')/255
    vali_data=[]
    for i in range(10):
        img = plt.imread('vali_data/img_'+str(i)+'.png')
        vali_data.append(img.tolist())
    vali_data=np.array(vali_data)
    vali_label=[]
    labels=[]
    with open('vali_data/labels.txt') as f:
        for l in f.readlines():
            labels.append(l.strip().split(','))
    for i in range(len(labels)):
        vali_label.append(int(labels[i][0]))
    return vali_data, vali_label
    