faceProbMatrix, notFaceProbMatrix, pyFace, pyNotFace = [],[],0,0


def train(train_data, train_label):
    global faceProbMatrix
    global notFaceProbMatrix
    global pyFace
    global pyNotFace

    allFaceData = createData(train_data)
    faceProbMatrix, notFaceProbMatrix, pyFace, pyNotFace = manualtrain(allFaceData, train_label)


def test(test_data, test_label):
    global faceProbMatrix
    global notFaceProbMatrix
    global pyFace
    global pyNotFace

    allFacesTest = createData(test_data)
    pred_labels=[]
    for i in range(len(allFacesTest)):
        singleLabel = getLabel(allFacesTest[i])
        pred_labels.append(singleLabel)
    c=0
    for i in range(len(test_label)):
        if test_label[i]==pred_labels[i]:
            c+=1
    print("Accuracy for Naive Bayes algorithm for Face classification:", (c/len(test_label))*100)

def getLabel(sampletest):
    global pyFace
    global pyNotFace
    prodFace = checkProbFace(sampletest)
    prodNotface = checkProbNonFace(sampletest)
    faceval, nonfaceval = 1, 1
    for i in range(len(prodFace)):
        faceval *= prodFace[i]
        nonfaceval *= prodNotface[i]
    faceval*=pyFace
    nonfaceval*=pyNotFace
    if faceval>nonfaceval:
        return 1
    else:
        return 0

#testing for a single image for face
def checkProbFace(sampletest):
    global faceProbMatrix
    prod=[]
    for i in range(len(faceProbMatrix)):
        if 0<=sampletest[i]<20:
            prod.append(faceProbMatrix[i][0])
        elif 20<=sampletest[i]<40:
            prod.append(faceProbMatrix[i][1])
        elif 40<=sampletest[i]<60:
            prod.append(faceProbMatrix[i][2])
        elif 60<=sampletest[i]<80:
            prod.append(faceProbMatrix[i][3])
        elif 80<=sampletest[i]<100:
            prod.append(faceProbMatrix[i][4])
        elif 100<=sampletest[i]<120:
            prod.append(faceProbMatrix[i][5])
        elif 120<=sampletest[i]<140:
            prod.append(faceProbMatrix[i][6])
        elif 140<=sampletest[i]<160:
            prod.append(faceProbMatrix[i][7])
        elif 160<=sampletest[i]<180:
            prod.append(faceProbMatrix[i][8])
        elif 180<=sampletest[i]<200:
            prod.append(faceProbMatrix[i][9])
        elif 200<=sampletest[i]<220:
            prod.append(faceProbMatrix[i][10])
        elif 220<=sampletest[i]<240:
            prod.append(faceProbMatrix[i][11])
        elif 240<=sampletest[i]<260:
            prod.append(faceProbMatrix[i][12])
        elif 260<=sampletest[i]<280:
            prod.append(faceProbMatrix[i][13])
    #print(prod)
    return prod

#testing for a single image for non face
def checkProbNonFace(sampletest):
    global notFaceProbMatrix
    prod=[]
    for i in range(len(notFaceProbMatrix)):
        if 0<=sampletest[i]<20:
            prod.append(notFaceProbMatrix[i][0])
        elif 20<=sampletest[i]<40:
            prod.append(notFaceProbMatrix[i][1])
        elif 40<=sampletest[i]<60:
            prod.append(notFaceProbMatrix[i][2])
        elif 60<=sampletest[i]<80:
            prod.append(notFaceProbMatrix[i][3])
        elif 80<=sampletest[i]<100:
            prod.append(notFaceProbMatrix[i][4])
        elif 100<=sampletest[i]<120:
            prod.append(notFaceProbMatrix[i][5])
        elif 120<=sampletest[i]<140:
            prod.append(notFaceProbMatrix[i][6])
        elif 140<=sampletest[i]<160:
            prod.append(notFaceProbMatrix[i][7])
        elif 160<=sampletest[i]<180:
            prod.append(notFaceProbMatrix[i][8])
        elif 180<=sampletest[i]<200:
            prod.append(notFaceProbMatrix[i][9])
        elif 200<=sampletest[i]<220:
            prod.append(notFaceProbMatrix[i][10])
        elif 220<=sampletest[i]<240:
            prod.append(notFaceProbMatrix[i][11])
        elif 240<=sampletest[i]<260:
            prod.append(notFaceProbMatrix[i][12])
        elif 260<=sampletest[i]<280:
            prod.append(notFaceProbMatrix[i][13])
    #print(prod)
    return prod

def manualtrain(allFaces, train_label):
    MIN_VALUE = 0.01
    tempFace = 0  # number of face items
    tempNotFace = 0  # number of non face items
    faceProbMatrix = [[0 for _ in range(14)] for _ in range(15)]
    notFaceProbMatrix = [[0 for _ in range(14)] for _ in range(15)]
    for i in train_label:
        if i == 1:
            tempFace += 1
        else:
            tempNotFace += 1
    pyFace = tempFace / len(train_label)
    pyNotFace = tempNotFace / len(train_label)
    # there are total 15 features(grids) and each grid contain 280 pixels for face data. Let us have range of 40 pixels and there will be total 7 different probability calculations
    #     allFaces = []
    # allFaces contains all the every data which has already done with feature extraction
    for i in range(len(allFaces)):
        if train_label[i] == 1:
            for j in range(len(allFaces[i])):
                if 0 <= allFaces[i][j] < 20:
                    faceProbMatrix[j][0] += 1
                elif 20 <= allFaces[i][j] < 40:
                    faceProbMatrix[j][1] += 1
                elif 40 <= allFaces[i][j] < 60:
                    faceProbMatrix[j][2] += 1
                elif 60 <= allFaces[i][j] < 80:
                    faceProbMatrix[j][3] += 1
                elif 80 <= allFaces[i][j] < 100:
                    faceProbMatrix[j][4] += 1
                elif 100 <= allFaces[i][j] < 120:
                    faceProbMatrix[j][5] += 1
                elif 120 <= allFaces[i][j] < 140:
                    faceProbMatrix[j][6] += 1
                elif 140 <= allFaces[i][j] < 160:
                    faceProbMatrix[j][7] += 1
                elif 160 <= allFaces[i][j] < 180:
                    faceProbMatrix[j][8] += 1
                elif 180 <= allFaces[i][j] < 200:
                    faceProbMatrix[j][9] += 1
                elif 200 <= allFaces[i][j] < 220:
                    faceProbMatrix[j][10] += 1
                elif 220 <= allFaces[i][j] < 240:
                    faceProbMatrix[j][11] += 1
                elif 240 <= allFaces[i][j] < 260:
                    faceProbMatrix[j][12] += 1
                elif 260 <= allFaces[i][j] < 280:
                    faceProbMatrix[j][13] += 1
        else:
            for j in range(len(allFaces[i])):
                if 0 <= allFaces[i][j] < 20:
                    notFaceProbMatrix[j][0] += 1
                elif 20 <= allFaces[i][j] < 40:
                    notFaceProbMatrix[j][1] += 1
                elif 40 <= allFaces[i][j] < 60:
                    notFaceProbMatrix[j][2] += 1
                elif 60 <= allFaces[i][j] < 80:
                    notFaceProbMatrix[j][3] += 1
                elif 80 <= allFaces[i][j] < 100:
                    notFaceProbMatrix[j][4] += 1
                elif 100 <= allFaces[i][j] < 120:
                    notFaceProbMatrix[j][5] += 1
                elif 120 <= allFaces[i][j] < 140:
                    notFaceProbMatrix[j][6] += 1
                elif 140 <= allFaces[i][j] < 160:
                    notFaceProbMatrix[j][7] += 1
                elif 160 <= allFaces[i][j] < 180:
                    notFaceProbMatrix[j][8] += 1
                elif 180 <= allFaces[i][j] < 200:
                    notFaceProbMatrix[j][9] += 1
                elif 200 <= allFaces[i][j] < 220:
                    notFaceProbMatrix[j][10] += 1
                elif 220 <= allFaces[i][j] < 240:
                    notFaceProbMatrix[j][11] += 1
                elif 240 <= allFaces[i][j] < 260:
                    notFaceProbMatrix[j][12] += 1
                elif 260 <= allFaces[i][j] < 280:
                    notFaceProbMatrix[j][13] += 1

    # converting the count for each feature into probabilities
    for x in range(len(faceProbMatrix)):
        for y in range(len(faceProbMatrix[0])):
            faceProbMatrix[x][y] = faceProbMatrix[x][y] / tempFace

    for x in range(len(notFaceProbMatrix)):
        for y in range(len(notFaceProbMatrix[0])):
            notFaceProbMatrix[x][y] = notFaceProbMatrix[x][y] / tempNotFace

    # print(faceProbMatrix)
    # print("**************************")
    # print(notFaceProbMatrix)
    return faceProbMatrix, notFaceProbMatrix, pyFace, pyNotFace

def createData(result):
    allFaceData=[]
    for i in range(len(result)):
        allFaceData.append(extractFeature(result[i]))
    return allFaceData




# takes one face ie List of one face and turns it into probability of '#' within grid and returns a list of all the gird probabilities
def extractFeature(oneface):
    faceFeature = []
    f00, f01, f02, f10, f11, f12, f20, f21, f22, f30, f31, f32, f40, f41, f42 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # there are total 15 grids for the image and we will be calculating probability of non spaces for each grid and return it as a single list
    # probability of non space for f00 feature
    for i in range(0, 14):
        for j in range(0, 20):
            if oneface[i][j] == 2:
                f00 += 1
    #     if f00>0:
    #         f00=f00/280*100

    # probability of non space for f01 feature
    for i in range(0, 14):
        for j in range(20, 40):
            if oneface[i][j] == 2:
                f01 += 1
    #     if f01>0:
    #         f01=f01/280*100

    # probability of non space for f02 feature
    for i in range(0, 14):
        for j in range(40, 60):
            if oneface[i][j] == 2:
                f02 += 1
    #     if f02>0:
    #         f02=f02/280*100

    # probability of non space for f10 feature
    for i in range(14, 28):
        for j in range(0, 20):
            if oneface[i][j] == 2:
                f10 += 1
    #     if f10>0:
    #         f10=f10/280*100

    # probability of non space for f11 feature
    for i in range(14, 28):
        for j in range(20, 40):
            if oneface[i][j] == 2:
                f11 += 1
    #     if f11>0:
    #         f11=f11/280*100

    # probability of non space for f12 feature
    for i in range(14, 28):
        for j in range(40, 60):
            if oneface[i][j] == 2:
                f12 += 1
    #     if f12>0:
    #         f12=f12/280*100

    # probability of non space for f20 feature
    for i in range(28, 42):
        for j in range(0, 20):
            if oneface[i][j] == 2:
                f20 += 1
    #     if f20>0:
    #         f20=f20/280*100

    # probability of non space for f21 feature
    for i in range(28, 42):
        for j in range(20, 40):
            if oneface[i][j] == 2:
                f21 += 1
    #     if f21>0:
    #         f21=f21/280*100

    # probability of non space for f22 feature
    for i in range(28, 42):
        for j in range(40, 60):
            if oneface[i][j] == 2:
                f22 += 1
    #     if f22>0:
    #         f22=f22/280*100

    # probability of non space for f30 feature
    for i in range(42, 56):
        for j in range(0, 20):
            if oneface[i][j] == 2:
                f30 += 1
    #     if f30>0:
    #         f30=f30/280*100

    # probability of non space for f31 feature
    for i in range(42, 56):
        for j in range(20, 40):
            if oneface[i][j] == 2:
                f31 += 1
    #     if f31>0:
    #         f31=f31/280*100

    # probability of non space for f32 feature
    for i in range(42, 56):
        for j in range(40, 60):
            if oneface[i][j] == 2:
                f32 += 1
    #     if f32>0:
    #         f32=f32/280*100

    # probability of non space for f40 feature
    for i in range(56, 70):
        for j in range(0, 20):
            if oneface[i][j] == 2:
                f40 += 1
    #     if f40>0:
    #         f40=f40/280*100

    # probability of non space for f41 feature
    for i in range(56, 70):
        for j in range(20, 40):
            if oneface[i][j] == 2:
                f41 += 1
    #     if f41>0:
    #         f41=f41/280*100

    # probability of non space for f42 feature
    for i in range(56, 70):
        for j in range(40, 60):
            if oneface[i][j] == 2:
                f42 += 1
    #     if f42>0:
    #         f42=f42/280*100
    faceFeature.append(f00)
    faceFeature.append(f01)
    faceFeature.append(f02)
    faceFeature.append(f10)
    faceFeature.append(f11)
    faceFeature.append(f12)
    faceFeature.append(f20)
    faceFeature.append(f21)
    faceFeature.append(f22)
    faceFeature.append(f30)
    faceFeature.append(f31)
    faceFeature.append(f32)
    faceFeature.append(f40)
    faceFeature.append(f41)
    faceFeature.append(f42)
    return faceFeature











