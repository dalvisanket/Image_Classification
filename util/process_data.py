import pandas as pd
import numpy as np
import math


def process_dataframe(path,data_height):
    #Load file into array
    file = [l[:-1] for l in open(path).readlines()]

    print("File length : ", len(file))

    #Create a Dataframe from the file
    df = pd.DataFrame(file, columns=['data'])

    #Split single column into image width columns
    df2 = df['data'].apply(lambda x: pd.Series(list(x)))

    #Replace all values with integers
    df2.replace(' ', 0, inplace=True)
    df2.replace('+', 1, inplace=True)
    df2.replace('#', 2, inplace=True)

    print("DataFrame shape : ",df2.shape)

    #Split one single df into number of images df
    df3 = np.asarray(np.array_split(df2.to_numpy(), math.floor(df2.shape[0] / data_height)))
    print("Final processed Dataframe shape : ",df3.shape)

    return df3



def process_label(path):

    # Load file into array
    file =[l[:-1] for l in open(path).readlines()]
    label = list(map(int,file))
    return np.array(label)

def merge_data(data1,data2):
    return np.concatenate((data1,data2))