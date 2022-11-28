import util.load_data as ld
import util.process_data as pd
import util.classifier_util as cu
import algorithm.perceptron as perceptron

if __name__ == '__main__':
    dataset = 'face'
    train_data,train_label,validation_data,validation_label,test_data,test_label = ld.load(dataset)
    weights = cu.generate_weights(dataset)
    total_train_data = pd.merge_data(train_data,validation_data)
    total_train_label = pd.merge_data(train_label,validation_label)

    print("****************** Started Training ******************")
    model = perceptron.train(train_data,train_label,weights,50)

    cu.find_accuracy(train_data,train_label,model)
    cu.find_accuracy(validation_data,validation_label,model)
    cu.find_accuracy(test_data,test_label,model)




