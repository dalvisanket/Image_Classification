import util.load_data as ld
import util.process_data as pd
import util.classifier_util as cu
import algorithm.perceptron as perceptron
import algorithm.svm as svm
import algorithm.naiveBayes as naiveBayes
import algorithm.naiveBayesDigit as naiveBayesDigit
import algorithm.naiveBayesDigitClassification as naiveBayesDigitClassification

if __name__ == '__main__':
    dataset = 'digit'
    train_data,train_label,validation_data,validation_label,test_data,test_label = ld.load(dataset)
    weights = cu.generate_weights(dataset)
    total_train_data = pd.merge_data(train_data,validation_data)
    print(total_train_data.shape)
    total_train_label = pd.merge_data(train_label,validation_label)
    print("****************** Please Select Algorithm ******************\n1. Perceptron\n2. Naive Bayes\n3. Support Vector Machine")
    choice = int(input("Enter choice\t"))
    if (choice == 1):
        print("****************** Started Training ******************")
        model = perceptron.train(train_data,train_label,weights,500)

        cu.find_accuracy(train_data,train_label,model)
        cu.find_accuracy(validation_data,validation_label,model)
        cu.find_accuracy(test_data,test_label,model)

    elif (choice == 2 and dataset == "face"):
        print("****************** Started Training ******************")
        naiveBayes.train(total_train_data, total_train_label)
        naiveBayes.test(test_data, test_label)

    # elif (choice == 2 and dataset == "digit"):
    #     print("****************** Started Training ******************")
    #     naiveBayesDigit.trainDigit(total_train_data, total_train_label)
    #     naiveBayesDigit.testDigit(test_data, test_label)


    elif (choice == 2 and dataset == "digit"):
        print("****************** Started Training ******************")
        naiveBayesDigitClassification.trainDigit(total_train_data, total_train_label)
        naiveBayesDigitClassification.testDigit(test_data, test_label)

    elif (choice == 3):
        print("****************** Started Training ******************")
        svm.train(dataset, total_train_data, total_train_label, test_data, test_label)





