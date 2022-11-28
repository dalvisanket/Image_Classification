import util.classifier_util as cu

def train(df,labels,weights,num_itr):

  for j in range(num_itr):
    for i in range(df.shape[0]):
      prediction = cu.classify(df[i],weights)
      actual = labels[i]

      if(prediction != actual):
        weights[prediction] = weights[prediction] - df[i]
        weights[actual] = weights[actual] + df[i]

  return weights