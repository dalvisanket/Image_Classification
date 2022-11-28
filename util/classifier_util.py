import numpy as np

def classify(df ,weights):
    weightSum = []
    for weight in weights:
        weightSum.append(np.sum(df*weight))

    return np.array(weightSum).argmax()


def find_accuracy(df,labels,weights):
    count = 0
    for i in range(df.shape[0]):
        prediction = classify(df[i],weights)
        actual = labels[i]
        if(prediction == actual):
          count = count + 1
    print("Correct predictions count : ",count,"/",df.shape[0])
    print("Accuracy Percentage : ",((count/df.shape[0])*100))

def generate_weights(dataset):
    count,h,w = ((2,70,60) if(dataset == 'face') else (10,28,28))
    weights = np.random.uniform(-1, 1, (count, h, w))
    print("Dimension of weight matrix : ",weights.shape)
    return weights

