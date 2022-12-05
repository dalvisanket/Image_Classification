import numpy as np
#******************TRAINING*****************OLD BUILD

zeroProbMatrix, oneProbMatrix, twoProbMatrix, threeProbMatrix, fourProbMatrix, fiveProbMatrix, sixProbMatrix, sevenProbMatrix, eightProbMatrix, nineProbMatrix = [],[],[],[],[],[],[],[],[],[]
pyZero, pyOne, pyTwo, pyThree, pyFour, pyFive, pySix, pySeven, pyEight, pyNine = 0,0,0,0,0,0,0,0,0,0

def trainDigit(train_data, train_label):
    global zeroProbMatrix,oneProbMatrix,twoProbMatrix,threeProbMatrix,fourProbMatrix,fiveProbMatrix,sixProbMatrix,sevenProbMatrix,eightProbMatrix,nineProbMatrix
    global pyZero, pyOne,pyTwo,pyThree,pyFour,pyFive,pySix,pySeven,pyEight,pyNine

    allDigitData = createData(train_data)
    # print(allDigitData[0])
    # print(allDigitData[1])
    # print(allDigitData[2])
    zeroProbMatrix, oneProbMatrix, twoProbMatrix, threeProbMatrix, fourProbMatrix, fiveProbMatrix, sixProbMatrix, sevenProbMatrix, eightProbMatrix, nineProbMatrix, pyZero, pyOne, pyTwo, pyThree, pyFour, pyFive, pySix, pySeven, pyEight, pyNine = manualtrain(allDigitData, train_label)
    # print(zeroProbMatrix)
    # print(oneProbMatrix)
    # # print(twoProbMatrix)
    # zeroProbMatrix = np.array(zeroProbMatrix1)
    # zeroProbMatrix[zeroProbMatrix == 0.0] = 0.01
    # oneProbMatrix = np.array(oneProbMatrix1)
    # oneProbMatrix[oneProbMatrix == 0.0] = 0.01
    # twoProbMatrix = np.array(twoProbMatrix1)
    # twoProbMatrix[twoProbMatrix == 0.0] = 0.01
    # threeProbMatrix = np.array(threeProbMatrix1)
    # threeProbMatrix[threeProbMatrix == 0.0] = 0.01
    # fourProbMatrix = np.array(fourProbMatrix1)
    # fourProbMatrix[fourProbMatrix == 0.0] = 0.01
    # fiveProbMatrix = np.array(fiveProbMatrix1)
    # fiveProbMatrix[fiveProbMatrix == 0.0] = 0.01
    # sixProbMatrix = np.array(sixProbMatrix1)
    # sixProbMatrix[sixProbMatrix == 0.0] = 0.01
    # sevenProbMatrix = np.array(sevenProbMatrix1)
    # sevenProbMatrix[sevenProbMatrix == 0.0] = 0.01
    # eightProbMatrix = np.array(eightProbMatrix1)
    # eightProbMatrix[eightProbMatrix == 0.0] = 0.01
    # nineProbMatrix = np.array(nineProbMatrix1)
    # nineProbMatrix[nineProbMatrix == 0.0] = 0.01


def manualtrain(allDigitData, train_label):
    MIN_VALUE = 0.01
    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = 0,0,0,0,0,0,0,0,0,0
    zeroProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    oneProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    twoProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    threeProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    fourProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    fiveProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    sixProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    sevenProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    eightProbMatrix = [[0 for _ in range(7)] for _ in range(16)]
    nineProbMatrix = [[0 for _ in range(7)] for _ in range(16)]


    for i in train_label:
        if i == 0:
            c0 += 1
        elif i == 1:
            c1 += 1
        elif i == 2:
            c2 += 1
        elif i == 3:
            c3 += 1
        elif i == 4:
            c4 += 1
        elif i == 5:
            c5 += 1
        elif i == 6:
            c6 += 1
        elif i == 7:
            c7 += 1
        elif i == 8:
            c8 += 1
        elif i == 9:
            c9 += 1
    pyZero = c0 / len(train_label)
    pyOne = c1 / len(train_label)
    pyTwo = c2 / len(train_label)
    pyThree = c3 / len(train_label)
    pyFour = c4 / len(train_label)
    pyFive = c5 / len(train_label)
    pySix = c6 / len(train_label)
    pySeven = c7 / len(train_label)
    pyEight = c8 / len(train_label)
    pyNine = c9 / len(train_label)
    # there are total 16 features(grids) and each grid contain 49 pixels for digit data. Let us have range of 7 pixels and there will be total 7 different probability calculations
    # allDigitData contains all the data which are already done with feature extraction
    for i in range(len(allDigitData)):
        if train_label[i] == 0:
            zeroProbMatrix = updateMatrix(zeroProbMatrix, allDigitData[i])
        elif train_label[i] == 1:
            oneProbMatrix = updateMatrix(oneProbMatrix, allDigitData[i])
        elif train_label[i] == 2:
            twoProbMatrix = updateMatrix(twoProbMatrix, allDigitData[i])
        elif train_label[i] == 3:
            threeProbMatrix = updateMatrix(threeProbMatrix, allDigitData[i])
        elif train_label[i] == 4:
            fourProbMatrix = updateMatrix(fourProbMatrix, allDigitData[i])
        elif train_label[i] == 5:
            fiveProbMatrix = updateMatrix(fiveProbMatrix, allDigitData[i])
        elif train_label[i] == 6:
            sixProbMatrix = updateMatrix(sixProbMatrix, allDigitData[i])
        elif train_label[i] == 7:
            sevenProbMatrix = updateMatrix(sevenProbMatrix, allDigitData[i])
        elif train_label[i] == 8:
            eightProbMatrix = updateMatrix(eightProbMatrix, allDigitData[i])
        elif train_label[i] == 9:
            nineProbMatrix = updateMatrix(nineProbMatrix, allDigitData[i])


    # converting the count for each feature into probabilities
    for x in range(len(zeroProbMatrix)):
        for y in range(len(zeroProbMatrix[0])):
            zeroProbMatrix[x][y] = zeroProbMatrix[x][y] / c0
    for x in range(len(oneProbMatrix)):
        for y in range(len(oneProbMatrix[0])):
            oneProbMatrix[x][y] = oneProbMatrix[x][y] / c1
    for x in range(len(twoProbMatrix)):
        for y in range(len(twoProbMatrix[0])):
            twoProbMatrix[x][y] = twoProbMatrix[x][y] / c2
    for x in range(len(threeProbMatrix)):
        for y in range(len(threeProbMatrix[0])):
            threeProbMatrix[x][y] = threeProbMatrix[x][y] / c3
    for x in range(len(fourProbMatrix)):
        for y in range(len(fourProbMatrix[0])):
            fourProbMatrix[x][y] = fourProbMatrix[x][y] / c4
    for x in range(len(fiveProbMatrix)):
        for y in range(len(fiveProbMatrix[0])):
            fiveProbMatrix[x][y] = fiveProbMatrix[x][y] / c5
    for x in range(len(sixProbMatrix)):
        for y in range(len(sixProbMatrix[0])):
            sixProbMatrix[x][y] = sixProbMatrix[x][y] / c6
    for x in range(len(sevenProbMatrix)):
        for y in range(len(sevenProbMatrix[0])):
            sevenProbMatrix[x][y] = sevenProbMatrix[x][y] / c7
    for x in range(len(eightProbMatrix)):
        for y in range(len(eightProbMatrix[0])):
            eightProbMatrix[x][y] = eightProbMatrix[x][y] / c8
    for x in range(len(nineProbMatrix)):
        for y in range(len(nineProbMatrix[0])):
            nineProbMatrix[x][y] = nineProbMatrix[x][y] / c9

    # print(faceProbMatrix)
    # print("**************************")
    # print(notFaceProbMatrix)
    return zeroProbMatrix, oneProbMatrix, twoProbMatrix, threeProbMatrix, fourProbMatrix, fiveProbMatrix, sixProbMatrix, sevenProbMatrix, eightProbMatrix, nineProbMatrix, pyZero, pyOne, pyTwo, pyThree, pyFour, pyFive, pySix, pySeven, pyEight, pyNine


#function to update the indiviual training matrix for each label
def updateMatrix(matrix, digitData):
    for j in range(len(digitData)):
        if 0 <= digitData[j] < 7:
            matrix[j][0] += 1
        elif 7 <= digitData[j] < 14:
            matrix[j][1] += 1
        elif 14 <= digitData[j] < 21:
            matrix[j][2] += 1
        elif 21 <= digitData[j] < 28:
            matrix[j][3] += 1
        elif 28 <= digitData[j] < 35:
            matrix[j][4] += 1
        elif 35 <= digitData[j] < 42:
            matrix[j][5] += 1
        elif 42 <= digitData[j] < 49:
            matrix[j][6] += 1
    return matrix







def createData(result):
    allDigitData=[]
    for i in range(len(result)):
        allDigitData.append(extractFeature(result[i]))
    return allDigitData

# takes one face ie List of one face and turns it into probability of '#' within grid and returns a list of all the gird probabilities
def extractFeature(oneDigit):
    digitFeature = []
    f00, f01, f02, f03, f10, f11, f12, f13, f20, f21, f22, f23, f30, f31, f32, f33 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # there are total 15 grids for the image and we will be calculating probability of non spaces for each grid and return it as a single list
    # probability of non space for f00 feature
    for i in range(0, 7):
        for j in range(0, 7):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f00 += 1

    for i in range(0, 7):
        for j in range(7, 14):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f01 += 1

    for i in range(0, 7):
        for j in range(14, 21):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f02 += 1

    for i in range(0, 7):
        for j in range(21, 28):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f03 += 1

    for i in range(7, 14):
        for j in range(0, 7):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f10 += 1

    for i in range(7, 14):
        for j in range(7, 14):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f11 += 1

    for i in range(7, 14):
        for j in range(14, 21):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f12 += 1

    for i in range(7, 14):
        for j in range(21, 28):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f13 += 1

    for i in range(14, 21):
        for j in range(0, 7):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f20 += 1

    for i in range(14, 21):
        for j in range(7, 14):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f21 += 1

    for i in range(14, 21):
        for j in range(14, 21):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f22 += 1

    for i in range(14, 21):
        for j in range(21, 28):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f23 += 1

    for i in range(21, 28):
        for j in range(0, 7):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f30 += 1

    for i in range(21, 28):
        for j in range(7, 14):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f31 += 1

    for i in range(21, 28):
        for j in range(14, 21):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f32 += 1

    for i in range(21, 28):
        for j in range(21, 28):
            if oneDigit[i][j] == 2 or oneDigit[i][j] == 1:
                f33 += 1
    digitFeature.append(f00)
    digitFeature.append(f01)
    digitFeature.append(f02)
    digitFeature.append(f03)
    digitFeature.append(f10)
    digitFeature.append(f11)
    digitFeature.append(f12)
    digitFeature.append(f13)
    digitFeature.append(f20)
    digitFeature.append(f21)
    digitFeature.append(f22)
    digitFeature.append(f23)
    digitFeature.append(f30)
    digitFeature.append(f31)
    digitFeature.append(f32)
    digitFeature.append(f33)
    return digitFeature



#******************TESTING*****************

def testDigit(test_data, test_label):
    global zeroProbMatrix, oneProbMatrix, twoProbMatrix, threeProbMatrix, fourProbMatrix, fiveProbMatrix, sixProbMatrix, sevenProbMatrix, eightProbMatrix, nineProbMatrix
    global pyZero, pyOne, pyTwo, pyThree, pyFour, pyFive, pySix, pySeven, pyEight, pyNine

    allDigitTest = createData(test_data)
    pred_labels=[]
    for i in range(len(allDigitTest)):
        singleLabel = getLabel(allDigitTest[i])
        pred_labels.append(singleLabel)
    c=0
    for i in range(len(test_label)):
        if test_label[i]==pred_labels[i]:
            c+=1
    print("Accuracy for Naive Bayes algorithm for Digit classification:", (c/len(test_label))*100)



def getLabel(sampletest):
    global pyZero, pyOne, pyTwo, pyThree, pyFour, pyFive, pySix, pySeven, pyEight, pyNine
    prod0 = checkProb(sampletest, zeroProbMatrix)
    prod1 = checkProb(sampletest, oneProbMatrix)
    prod2 = checkProb(sampletest, twoProbMatrix)
    prod3 = checkProb(sampletest, threeProbMatrix)
    prod4 = checkProb(sampletest, fourProbMatrix)
    prod5 = checkProb(sampletest, fiveProbMatrix)
    prod6 = checkProb(sampletest, sixProbMatrix)
    prod7 = checkProb(sampletest, sevenProbMatrix)
    prod8 = checkProb(sampletest, eightProbMatrix)
    prod9 = checkProb(sampletest, nineProbMatrix)

    # val0, val1, val2, val3, val4, val5, val6, val7, val8, val9 = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    val0, val1, val2, val3, val4, val5, val6, val7, val8, val9 = prod0[0],prod1[0],prod2[0],prod3[0],prod4[0],prod5[0],prod6[0],prod7[0],prod8[0],prod9[0]
    for i in range(1, len(prod0)):
        if val0>0:
            val0 *= prod0[i]
        if val1 > 0:
            val1 *= prod1[i]
        if val2 > 0:
            val2 *= prod2[i]
        if val3 > 0:
            val3 *= prod3[i]
        if val4 > 0:
            val4 *= prod4[i]
        if val5 > 0:
            val5 *= prod5[i]
        if val6 > 0:
            val6 *= prod6[i]
        if val7 > 0:
            val7 *= prod7[i]
        if val8 > 0:
            val8 *= prod8[i]
        if val9 > 0:
            val9 *= prod9[i]

    val0 *= pyZero
    val1 *= pyOne
    val2 *= pyTwo
    val3 *= pyThree
    val4 *= pyFour
    val5 *= pyFive
    val6 *= pySix
    val7 *= pySeven
    val8 *= pyEight
    val9 *= pyNine
    maxval = max(val1,val2, val3,val4,val5,val6,val7,val8,val9,val0)
    # print(maxval)
    if maxval == val0:
        return 0
    elif maxval == val1:
        return 1
    elif maxval == val2:
        return 2
    elif maxval == val3:
        return 3
    elif maxval == val4:
        return 4
    elif maxval == val5:
        return 5
    elif maxval ==val6:
        return 6
    elif maxval == val7:
        return 7
    elif maxval == val8:
        return 8
    elif maxval == val9:
        return 9



def checkProb(sampletest, probMatrix):
    prod = []
    for i in range(len(sampletest)):
        if 0 <= sampletest[i] < 7:
            prod.append(probMatrix[i][0])
        elif 7 <= sampletest[i] < 14:
            prod.append(probMatrix[i][1])
        elif 14 <= sampletest[i] < 21:
            prod.append(probMatrix[i][2])
        elif 21 <= sampletest[i] < 28:
            prod.append(probMatrix[i][3])
        elif 28 <= sampletest[i] < 35:
            prod.append(probMatrix[i][4])
        elif 35 <= sampletest[i] < 42:
            prod.append(probMatrix[i][5])
        elif 42 <= sampletest[i] < 49:
            prod.append(probMatrix[i][6])
    # print(prod)
    return prod