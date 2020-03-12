import numpy as np
import matplotlib.pyplot as plt
import os, csv, random, collections
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import hamming_loss
from sklearn.svm import SVC, LinearSVC
from imblearn.over_sampling import SMOTE

class hw7():
    #5036 + 2159
    trainData = np.zeros((5036, 22), dtype=float)
    testData = np.zeros((2159, 22), dtype=float)
    trainLabel1, trainLabel2, trainLabel3 = [], [], []
    testLabel1, testLabel2, testLabel3 = [], [], []
    yTrain = np.zeros((5036, 3), dtype=float)
    yTest = np.zeros((2159, 3), dtype=float)
    dicLabel1 = ['Bufonidae','Dendrobatidae','Hylidae','Leptodactylidae']
    dicLabel2 = ['Adenomera','Ameerega','Dendropsophus','Hypsiboas','Leptodactylus',
                 'Osteocephalus','Rhinella','Scinax']
    dicLabel3 = ['AdenomeraAndre','AdenomeraHylaedactylus','Ameeregatrivittata','HylaMinuta',
                 'HypsiboasCinerascens','HypsiboasCordobae','LeptodactylusFuscus','OsteocephalusOophagus',
                 'Rhinellagranulosa','ScinaxRuber']
    #RBF + L2               bestModel1 list
    bestModel1 = [[0.007742083697420191, 7.38905609893065, 2.0], 
                  [0.009928328046201423, 7.38905609893065, 1.6], 
                  [0.009306536713306498, 20.085536923187668, 1.9000000000000001]]
    #Linear + L1            bestModel2 list
    bestModel2 = [[0.06512631728995809, 7.38905609893065], 
                  [0.05181075296983158, 54.598150033144236], 
                  [0.04389161273429434, 7.38905609893065]]
    #Linear + L1 + SMOTE    bestModel3 list
    bestModel3 = [[0.05626559922162527, 2.718281828459045], 
                  [0.04733816320645905, 148.4131591025766], 
                  [0.041852988580638524, 20.085536923187668]]
    #RBF + L2 + SMOTE       bestModel4 list
    bestModel4 =  [[0.0025281556705201084, 148.4131591025766, 1.3000000000000003], 
                   [0.0007783975659229211, 20.085536923187668, 2.0], 
                   [0.0003305785123966942, 148.4131591025766, 2.0]]
    #Linear + L1 + Classifier Chain chainModel list
    chainModel = [[0.06512631728995809, 7.38905609893065], 
                  [0.0113109507997832, 148.4131591025766], 
                  [0.003571162055746495, 148.4131591025766]]
    def __init__(self):
        try:
            os.chdir('/Users/chengyaowang/Desktop/USC/EE559/Homework/Homework7/Anuran Calls (MFCCs)')
        except:
            os.chdir("F://University of Southern California//EE559//Homework//Homework7//Anuran Calls (MFCCs)")
        #self.preprocessData()
        self.readData()
        self.labelInit()
        print( 'Data Initiation Completed' )
    def preprocessData(self):
        #Store the list of data used for training in a csv file
        trainList = np.sort( random.sample(range(7195), 5036) )
        train, test, rawData = [], [], []
        #Read the Data
        with open('frogs_Cleaned.csv') as csv_file:
            csv_reader=csv.reader(csv_file, delimiter=',')
            for rowPointer in csv_reader:
                if rowPointer[0] == 'MFCCs_ 1':
                    continue
                rawData.append(rowPointer)
        csv_file.close()
        for iter in range(7195):
            train.append(rawData[iter]) if iter in trainList else test.append(rawData[iter])
        with open('trainData.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(train)
        writeFile.close()
        with open('testData.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(test)
        writeFile.close()
    def readData(self):
        #Read the Train Data
        with open('trainData.csv') as csv_file:
            csv_reader=csv.reader(csv_file, delimiter=',')
            rowCnt = 0
            for rowPointer in csv_reader:
                self.trainData[rowCnt, :] = np.array( rowPointer[:22])
                self.trainLabel1.append( rowPointer[22] )
                self.trainLabel2.append( rowPointer[23] )
                self.trainLabel3.append( rowPointer[24] )
                rowCnt += 1
        csv_file.close()
        #Read the Test Data
        with open('testData.csv') as csv_file:
            csv_reader=csv.reader(csv_file, delimiter=',')
            rowCnt = 0
            for rowPointer in csv_reader:
                self.testData[rowCnt, :] = np.array( rowPointer[:22])
                self.testLabel1.append( rowPointer[22] )
                self.testLabel2.append( rowPointer[23] )
                self.testLabel3.append( rowPointer[24] )
                rowCnt += 1
        csv_file.close()
    def labelInit(self):
        le1 = preprocessing.LabelEncoder().fit(self.dicLabel1)
        le2 = preprocessing.LabelEncoder().fit(self.dicLabel2)
        le3 = preprocessing.LabelEncoder().fit(self.dicLabel3)
        self.yTrain[:, 0] = le1.transform(self.trainLabel1)
        self.yTrain[:, 1] = le2.transform(self.trainLabel2)
        self.yTrain[:, 2] = le3.transform(self.trainLabel3)
        self.yTest[:, 0]  = le1.transform(self.testLabel1)
        self.yTest[:, 1]  = le2.transform(self.testLabel2)
        self.yTest[:, 2]  = le3.transform(self.testLabel3)
    #Output the Results into a text File
    def resultOutput(self):
        os.chdir('/Users/chengyaowang/Desktop/USC/EE559/Homework/Homework7')
        #with open('results.text') as txt_file:
    #Prepare Data for SVM CV
    #Arguments: ylabel
    #Returns: 10 * NumOfInstances/10 * 22 Data Matrix, 10 * NumOfInstances/10 * 1 Data Matrix
    def dataForCV(self, ylabel):
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        #The Train/Test Indices for Label 1, 2, 3
        skf.get_n_splits(self.trainData, ylabel)
        dataReturn, labelReturn = [], []
        for _, testIndex in skf.split(self.trainData, ylabel):
            dataReturn.append( self.trainData[np.array(testIndex)] )
            labelReturn.append( ylabel[np.array(testIndex)] )
        return np.asarray(dataReturn), np.asarray(labelReturn)
    # Gaussian Kernel, OvR, 10-fold CV, SVM, Hamming Loss
    def svmPerform(self):
        penaltyList = np.exp( range(-3, 6))
        sigmaList = np.arange(0.1, 2.1, 0.1)
        #Choose a Label Among Three
        for labelCnt in range(3):
        #Get the Folded Data from dataForCV()
            X, y = self.dataForCV(self.yTrain[:, labelCnt])
        #Go Through Every Combination of HyperParameters
            handshakeCnt = 0
            for C in penaltyList:
                for sigma in sigmaList:
                    coreEstimator = SVC(C=C, kernel='rbf', gamma=sigma)
                    OvRestimator = OneVsRestClassifier(coreEstimator)
        #Initiate Train/Test Data, foldChoose is used for Validation
                    hammingScore = 0
                    for foldChoose in range(10):
                        i = int(foldChoose == 0)
                        CVtraindata, CVtrainlabel = X[i], y[i]
                        CVtestdata, CVtestlabel = X[foldChoose], y[foldChoose]
                        while i < 9:
                            i += 1
                            if i == foldChoose:
                                continue
                            CVtraindata  = np.concatenate( (CVtraindata,  X[i]), axis=0)
                            CVtrainlabel = np.concatenate( (CVtrainlabel, y[i]), axis=0)
        #Fit the data and get Hamming Score
                        OvRestimator.fit(CVtraindata, CVtrainlabel)
                        hammingScore += hamming_loss( CVtestlabel, OvRestimator.predict(CVtestdata) )/10
        #Check if is the best Score
                    if hammingScore <= self.bestModel1[labelCnt][0]:
                        self.bestModel1[labelCnt] = [hammingScore, C, sigma]
                    handshakeCnt += 1
                    print( 'HandShake Count:', handshakeCnt )
            print( 'One Label Completed' )
        print( 'Best Model with RBF L2 Penalized\n', self.bestModel1 )
    # Linear Kernel, OvR, 10-fold CV, SVM, Hamming Loss
    def l1Penalized(self):
        penaltyList = np.exp( range(-3, 6) )
        #Choose a Label Among Three
        for labelCnt in range(3):
        #Get the Folded Data from dataForCV()
            X, y = self.dataForCV(self.yTrain[:, labelCnt])
            handshakeCnt = 0
        #Go Through Every Combination of HyperParameters
            for C in penaltyList:
                coreEstimator = LinearSVC(penalty = 'l1', C = C, dual=False, max_iter=100000)
                OvRestimator = OneVsRestClassifier(coreEstimator)
        #Initiate Train/Test Data, foldChoose is used for Validation
                hammingScore = 0
                for foldChoose in range(10):
                    i = int(foldChoose == 0)
                    CVtraindata, CVtrainlabel = X[i], y[i]
                    CVtestdata, CVtestlabel = X[foldChoose], y[foldChoose]
                    while i < 9:
                        i += 1
                        if i == foldChoose:
                            continue
                        CVtraindata  = np.concatenate( (CVtraindata,  X[i]), axis=0)
                        CVtrainlabel = np.concatenate( (CVtrainlabel, y[i]), axis=0)
        #Fit the data and get Hamming Score
                    OvRestimator.fit(CVtraindata, CVtrainlabel)
                    hammingScore += hamming_loss( CVtestlabel, OvRestimator.predict(CVtestdata) )/10
        #Check if is the best Score
                if hammingScore <= self.bestModel2[labelCnt][0]:
                    self.bestModel2[labelCnt] = [hammingScore, C]
                handshakeCnt += 1
                print( 'HandShake Count:', handshakeCnt )
            print( 'One Label Completed' )
        print( 'Best Model with Linear L1 Penalized\n', self.bestModel2 )
    # SMOTE implementation
    def smotePerform(self, ylabel):
        smModel = SMOTE()
        resampledData, resampledLabel = smModel.fit_resample(self.trainData, ylabel)
        print('Before Smote:', collections.Counter( np.ndarray.tolist(ylabel) ) )
        print('After Smote:',  collections.Counter( np.ndarray.tolist(resampledLabel) ) )
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        #The Train/Test Indices for Label 1, 2, 3
        skf.get_n_splits(resampledData, resampledLabel)
        dataReturn, labelReturn = [], []
        for _, testIndex in skf.split(resampledData, resampledLabel):
            dataReturn.append( resampledData[np.array(testIndex)] )
            labelReturn.append( resampledLabel[np.array(testIndex)] )
        return np.asarray(dataReturn), np.asarray(labelReturn)
    # Linear Kernel, OvR, 10-fold CV, SVM, Hamming Loss, SMOTE
    def l1WithSMOTE(self):
        penaltyList = np.exp( range(-3, 6) )
        #Choose a Label Among Three
        for labelCnt in range(3):
        #Get the Folded Data from dataForCV()
            X, y = self.smotePerform(self.yTrain[:, labelCnt])
            handshakeCnt = 0
        #Go Through Every Combination of HyperParameters
            for C in penaltyList:
                coreEstimator = LinearSVC(penalty = 'l1', C = C, dual=False, max_iter=100000)
                OvRestimator = OneVsRestClassifier(coreEstimator)
        #Initiate Train/Test Data, foldChoose is used for Validation
                hammingScore = 0
                for foldChoose in range(10):
                    i = int(foldChoose == 0)
                    CVtraindata, CVtrainlabel = X[i], y[i]
                    CVtestdata, CVtestlabel = X[foldChoose], y[foldChoose]
                    while i < 9:
                        i += 1
                        if i == foldChoose:
                            continue
                        CVtraindata  = np.concatenate( (CVtraindata,  X[i]), axis=0)
                        CVtrainlabel = np.concatenate( (CVtrainlabel, y[i]), axis=0)
        #Fit the data and get Hamming Score
                    OvRestimator.fit(CVtraindata, CVtrainlabel)
                    hammingScore += hamming_loss( CVtestlabel, OvRestimator.predict(CVtestdata) )/10
        #Check if is the best Score
                if hammingScore <= self.bestModel3[labelCnt][0]:
                    self.bestModel3[labelCnt] = [hammingScore, C]
                handshakeCnt += 1
                print( 'HandShake Count:', handshakeCnt )
            print( 'One Label Completed' )
        print( 'Best Model with Linear L1 Penalized SMOTE\n', self.bestModel3 )
    #Data for Classifer Chain, Array Concatenate And Train-Validation Split
    def dataForChain(self, labelIndex):
        #Concatenate Labels into feature Matrix
        lb = LabelBinarizer().fit( [0, 1, 2, 3] )
        plugInLabel = np.array( lb.transform( self.yTrain[:, 0] ) )
        X  = np.concatenate( (self.trainData, plugInLabel) , axis=1)
        #If the Data is used for the last model, need to plug in two features 
        if labelIndex == 1:
            lb = LabelBinarizer().fit( [0, 1, 2, 3, 4, 5, 6, 7] )
            plugInLabel = np.array( lb.transform( self.yTrain[:, 1] ) )
            X  = np.concatenate( (X, plugInLabel) , axis=1)
        #Do Cross Validation
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        #The Train/Test Indices for Label 1, 2, 3
        skf.get_n_splits( X, self.yTrain[:, labelIndex+1])
        dataReturn, labelReturn = [], []
        for _, testIndex in skf.split( X, self.yTrain[:, labelIndex+1]):
            dataReturn.append( X[np.array(testIndex)] )
            labelReturn.append( self.yTrain[:, labelIndex+1][np.array(testIndex)] )
        return np.asarray(dataReturn), np.asarray(labelReturn)
    # Linear Kernel, OvR, 10-fold CV, SVM, Hamming Loss, Classifier Chain
    #Chain Order: Family, Genus, Species
    def classifierChain(self):
        penaltyList = np.exp( range(-3, 6) )
        #Choose a Label Among The Last two
        for labelCnt in [0, 1]:
        #Get the Folded Data from dataForCV()
            X, y = self.dataForChain( labelCnt )
            handshakeCnt = 0 
        #Go Through Every Combination of HyperParameters
            for C in penaltyList:
                coreEstimator = LinearSVC(penalty = 'l1', C = C, dual=False, max_iter=1000000)
                OvRestimator  = OneVsRestClassifier(coreEstimator)
        #Initiate Train/Test Data, foldChoose is used for Validation
                hammingScore = 0
                for foldChoose in range(10):
                    i = int(foldChoose == 0)
                    CVtraindata, CVtrainlabel = X[i], y[i]
                    CVtestdata, CVtestlabel = X[foldChoose], y[foldChoose]
                    while i < 9:
                        i += 1
                        if i == foldChoose:
                            continue
                        CVtraindata  = np.concatenate( (CVtraindata,  X[i]), axis=0)
                        CVtrainlabel = np.concatenate( (CVtrainlabel, y[i]), axis=0)
        #Fit the data and get Hamming Score
                    OvRestimator.fit(CVtraindata, CVtrainlabel)
                    hammingScore += hamming_loss( CVtestlabel, OvRestimator.predict(CVtestdata) )/10
        #Check if is the best Score
                if hammingScore <= self.bestModel2[ labelCnt+1 ][0]:
                    self.chainModel[ labelCnt+1 ] = [hammingScore, C]
                handshakeCnt += 1
                print( 'HandShake:', handshakeCnt )
            print( 'One Label Completed' )
        print( 'Best Chain Classifier Model Under HammingScore\n', self.chainModel )
    #Test the Optimal Classifier Achieved Using Test Data
    def optimalTest(self):
        # Gaussian Kernel, OvR, 10-fold CV, SVM, Hamming Loss
        print ('RBF Kernel + L2 Penalty')
        # Label 1: Families
        coreEstimator = SVC(C=float(self.bestModel1[0][1]), kernel='rbf', gamma=float(self.bestModel1[0][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 0])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 0] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 0] )
        # Label 2: Genus:
        coreEstimator = SVC(C=float(self.bestModel1[1][1]), kernel='rbf', gamma=float(self.bestModel1[1][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 1])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 1] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 1] )
        # Label 3: Species:
        coreEstimator = SVC(C=float(self.bestModel1[2][1]), kernel='rbf', gamma=float(self.bestModel1[2][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 2])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 2] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 2] )
        # Linear Kernel, OvR, 10-fold CV, SVM, Hamming Loss
        print ('Linear Kernel + L1 Penalty')
        # Label 1: Families
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel2[0][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 0])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 0] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 0] )
        # Label 2: Genus:
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel2[1][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 1])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 1] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 1] )
        # Label 3: Species:
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel2[2][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 2])
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 2] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 2] )
        #Create SMOTE data
        print ('SMOTE:')
        smModel = SMOTE()
        resampledData1, resampledLabel1 = smModel.fit_resample(self.trainData, self.yTrain[:, 0])
        resampledData2, resampledLabel2 = smModel.fit_resample(self.trainData, self.yTrain[:, 1])
        resampledData3, resampledLabel3 = smModel.fit_resample(self.trainData, self.yTrain[:, 2])
        #Linear + L1 + SMOTE    bestModel3 list
        # Label 1: Families
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel3[0][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData1, resampledLabel1)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 0] )
        self.printresult( OvRestimator.predict(resampledData1), resampledLabel1 )
        # Label 2: Genus:
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel3[1][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData2, resampledLabel2)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 1] )
        self.printresult( OvRestimator.predict(resampledData2), resampledLabel2 )
        # Label 3: Species:
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.bestModel3[2][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData3, resampledLabel3)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 2] )
        self.printresult( OvRestimator.predict(resampledData3), resampledLabel3 )
        #RBF + L2 + SMOTE    bestModel4 list
        # Label 1: Families
        coreEstimator = SVC(C=float(self.bestModel4[0][1]), kernel='rbf', gamma=float(self.bestModel4[0][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData1, resampledLabel1)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 0] )
        self.printresult( OvRestimator.predict(resampledData1), resampledLabel1 )
        # Label 2: Genus:
        coreEstimator = SVC(C=float(self.bestModel4[0][1]), kernel='rbf', gamma=float(self.bestModel4[0][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData2, resampledLabel2)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 1] )
        self.printresult( OvRestimator.predict(resampledData2), resampledLabel2 )
        # Label 3: Species:
        coreEstimator = SVC(C=float(self.bestModel4[0][1]), kernel='rbf', gamma=float(self.bestModel4[0][2]))
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(resampledData3, resampledLabel3)
        self.printresult( OvRestimator.predict(self.testData), self.yTest[:, 2] )
        self.printresult( OvRestimator.predict(resampledData3), resampledLabel3 )
        #Linear + L1 + Classifier Chain chainModel list
        # First Chain, Feature -> Label 1: Families
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.chainModel[0][1]), dual=False, max_iter=100000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        OvRestimator.fit(self.trainData, self.yTrain[:, 0])
        label1Prediction = OvRestimator.predict(self.testData)
        self.printresult( label1Prediction, self.yTest[:, 0] )
        self.printresult( OvRestimator.predict(self.trainData), self.yTrain[:, 0] )
        # Second Chain, Feature + Label 1 -> Label 2: Genus
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.chainModel[1][1]), dual=False, max_iter=1000000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        lb = LabelBinarizer().fit( [0, 1, 2, 3] )
        plugInLabel_Train = np.array( lb.transform( self.yTrain[:, 0] ) )
        plugInLabel_Test  = np.array( lb.transform( label1Prediction ) )
        augmentedTrainData = np.concatenate( (self.trainData, plugInLabel_Train), axis=1)
        augmentedTestData  = np.concatenate( (self.testData, plugInLabel_Test), axis=1)
        OvRestimator.fit( augmentedTrainData , self.yTrain[:, 1])
        label2Prediction = OvRestimator.predict( augmentedTestData  )
        self.printresult( OvRestimator.predict( augmentedTrainData  ) , self.yTrain[:, 1] )
        self.printresult( label2Prediction, self.yTest[:, 1] )
        # Third Chain, Feature + Label 1 + Label 2 -> Label 3: Species
        coreEstimator = LinearSVC(penalty = 'l1', C = float(self.chainModel[2][1]), dual=False, max_iter=1000000)
        OvRestimator = OneVsRestClassifier(coreEstimator)
        lb = LabelBinarizer().fit( [0, 1, 2, 3, 4, 5, 6, 7] )
        plugInLabel_Train = np.array( lb.transform( self.yTrain[:, 1] ) )
        plugInLabel_Test  = np.array( lb.transform( label1Prediction ) )
        augmentedTrainData = np.concatenate( (augmentedTrainData, plugInLabel_Train), axis=1)
        augmentedTestData  = np.concatenate( (augmentedTestData, plugInLabel_Test), axis=1)
        OvRestimator.fit( augmentedTrainData , self.yTrain[:, 2])
        label3Prediction = OvRestimator.predict( augmentedTestData  )
        self.printresult( OvRestimator.predict( augmentedTrainData  ) , self.yTrain[:, 2] )
        self.printresult( label3Prediction, self.yTest[:, 2] )
    #Print Function of Optimal Test
    def printresult(self, predval, trueval):
        preddata = [0.0] *11
        truedata = [0.0] *11
        for i in range( predval.shape[0] ):
            preddata[ int(predval[i]) ] += 1
            truedata[ int(trueval[i]) ] += 1
        print ( preddata, '\n', truedata)
        print ( hamming_loss( predval, trueval) )



SVMObject = hw7()
#SVMObject.svmPerform()
#SVMObject.l1Penalized()
#SVMObject.l1WithSMOTE()
#SVMObject.classifierChain()
SVMObject.optimalTest()
