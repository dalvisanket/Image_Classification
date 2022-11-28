import util.process_data as pd

def load(dataset):
    train_data = None
    train_label = None

    validation_data = None
    validation_label = None

    test_data = None
    test_label = None

    if(dataset == 'face'):
        print("********************* Loading Face Data *********************")
        train_data = pd.process_dataframe('dataset/face/facedatatrain',70)
        train_label = pd.process_label('dataset/face/facedatatrainlabels')

        validation_data = pd.process_dataframe('dataset/face/facedatavalidation',70)
        validation_label = pd.process_label('dataset/face/facedatavalidationlabels')

        test_data = pd.process_dataframe('dataset/face/facedatatest',70)
        test_label = pd.process_label('dataset/face/facedatatestlabels')

    elif(dataset == 'digit'):
        print("********************* Loading Digit Data *********************")
        train_data = pd.process_dataframe('dataset/digit/trainingimages',28)
        train_label = pd.process_label('dataset/digit/traininglabels')

        validation_data = pd.process_dataframe('dataset/digit/validationimages',28)
        validation_label = pd.process_label('dataset/digit/validationlabels')

        test_data = pd.process_dataframe('dataset/digit/testimages',28)
        test_label = pd.process_label('dataset/digit/testlabels')

    return train_data,train_label,validation_data,validation_label,test_data,test_label