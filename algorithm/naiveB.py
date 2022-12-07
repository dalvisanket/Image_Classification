import numpy as np
import math

feature_dimension = 4
total_features = 49
def train(df,label):

    df_feature_split = split_image_into_features(df)


    feature_prob = np.full((10,total_features,feature_dimension),0.01,dtype=float)

    label_count = np.full((10),0,dtype=int)


    for image in range(0, df_feature_split.shape[0]):
        feature_prob,label_count = count_image_feature_prob(df_feature_split[image],label[image],label_count,feature_prob)

    for count in range(label_count.shape[0]):
        feature_prob[count] = feature_prob[count]/label_count[count]

    prob_of_digit = label_count/df.shape[0]

    accuracy(df,label,feature_prob,prob_of_digit)
    return feature_prob,prob_of_digit



def split_image_into_features(df):
    df_feature_split = []

    for image in df:
        df_feature_split.append(np.asarray(
            [image[x:x + feature_dimension, y:y + feature_dimension] for x in range(0, image.shape[0], feature_dimension) for y in range(0, image.shape[1], feature_dimension)]))

    return np.asarray(df_feature_split)


def count_image_feature_prob(image,label,label_count, feature_prob):
    for feature in range(0,image.shape[0]):
        feature_count = (image[feature] == 1).sum() + (image[feature] == 2).sum()
        index  = math.floor(feature_count/feature_dimension)
        if(index == feature_prob.shape[2]):
            index = index - 1

        feature_prob[label][feature][index] = feature_prob[label][feature][index] + 1
    label_count[label] = label_count[label] + 1
    return feature_prob,label_count

def classify(image,feature_prob,prob_of_digit):

    image_feature_count_index = []
    for feature in range(0,image.shape[0]):
        feature_count = (image[feature] == 1).sum() + (image[feature] == 2).sum()
        index  = math.floor(feature_count/feature_dimension)
        if(index == feature_prob.shape[2]):
            index = index - 1

        image_feature_count_index.append(index)

    digit_classify_prob = []
    for digit_feature_prob in range(0,feature_prob.shape[0]):
        prob_product = 1
        for index in range(0,feature_prob[0].shape[0]):
            prob_product = prob_product * feature_prob[digit_feature_prob][index][image_feature_count_index[index]]
        digit_classify_prob.append(prob_product*prob_of_digit[digit_feature_prob])
    return np.asarray(digit_classify_prob).argmax()


def accuracy(df,label,feature_prob,prob_of_digit):
    df_feature_split = split_image_into_features(df)
    count = 0;
    for image in range(0, df_feature_split.shape[0]):
        if (classify(df_feature_split[image], feature_prob, prob_of_digit) == label[image]):
            count = count + 1

    print('Accuracy : ',(count / df.shape[0])*100)
