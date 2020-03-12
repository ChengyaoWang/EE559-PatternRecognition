# -*- coding: utf-8 -*-
"""
Created on Jun Sat 13:52:59 2019
Homework3 for EE559
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import datasets
from scipy import interp
import scipy.stats as stat
import statsmodels.formula.api as sm
from itertools import cycle
#Important note: Dataset8 of Sitting missing value of t=13500
#Filled with arithmic average of t=13250 and t=13750
class hw3(object):
    #Some Information about data sets, and Configurations of the Projects
    raw_dataset=np.zeros((88,480,6), dtype=float)
    maxList=list(['min', 'max', 'mean', 'median', 'std', '25percentile', '75percentile'])
    testList=[1, 2, 8, 9, 14, 15, 16, 29, 30, 31, 44, 45, 46, 59, 60, 61, 74, 75, 76]
    dirList=list(['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking'])
    selectedList=list(['mean', 'std', 'max'])
    NumTrainClass=[9, 12, 12, 12, 12, 12]
    NumTestClass=[4, 3, 3, 3, 3, 3]
    #Some Optimal Values found
    bestL_1000=4
    bestL_7=19
    bestl1=1
    bestAlpha=0.36787944
    bestMultiAlpha=148.4131591025766
    bestfeatureList_1000=[1, 4, 16, 21, 22, 23, 26, 40, 50, 56, 64, 65, 67]
    bestfeatureList_7=[10, 13, 191, 192, 305, 306]
    #Data Set After data preprocessing
    trueTrainLabel=np.zeros(69)
    trueTestLabel=np.zeros(19)
    trueMultiTrainLabel=np.zeros(69)
    trueMultiTestLabel=np.zeros(19)
    def __init__(self, l):
        self.l=l
        self.readData()
        self.timeSeriesDivision()
        self.featureSelection()
        self.truelabelInit()
        self.dataResize()
        
        print "Initialization Complete."
    def truelabelInit(self):
        for i in range(69):
            if i<9:
                self.trueTrainLabel[i]=1
        for i in range(19):
            if i<4:
                self.trueTestLabel[i]=1
        testCnt=0
        trainCnt=0
        for i in range(6):
            for pnt in range(self.NumTestClass[i]):
                self.trueMultiTestLabel[testCnt]=i
                testCnt+=1
            for pnt in range(self.NumTrainClass[i]):
                self.trueMultiTrainLabel[trainCnt]=i
                trainCnt+=1
    def readOneCsv(self, dir, fileName, numInstance):
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/AReM/"+dir)
        read_latch=0
        row_cnt=0
        with open(fileName+".csv") as csv_file:
            csv_reader=csv.reader(csv_file, delimiter=',')
            for rowPointer in csv_reader:
                if (rowPointer[0]=='0') & (~read_latch):
                    read_latch=1
                if read_latch:
                    temp=np.array(rowPointer, dtype=float)
                    self.raw_dataset[numInstance, row_cnt, :]=temp[-6:]
                    row_cnt+=1
    #Input Raw Dataset
    def readData(self):
        fileName_base="dataset"
        instance_cnt=0
        for dir_iter in self.dirList:
            for file_iter in range(1,16):
                try:
                    self.readOneCsv(dir_iter, fileName_base+str(file_iter), instance_cnt)
                    #print dir_iter,"Dataset",file_iter,"Input Completed"
                    instance_cnt+=1
                except:
                    pass
        print "Data Reading Complete"
    def featureExtract(self, arrayIn):
        out=np.zeros((7), dtype=float)
        out[0]=min(arrayIn)
        out[1]=max(arrayIn)
        out[2]=np.mean(arrayIn)
        out[3]=np.median(arrayIn)
        out[4]=np.std(arrayIn)
        out[5]=np.percentile(arrayIn,25)
        out[6]=np.percentile(arrayIn,75)
        #print "Feature Extraction Completed"
        return out
    def featureSelection(self):
        self.divTrainSet_Sorted=np.zeros((6*self.l, 69, len(self.selectedList)), dtype=float)
        self.divTestSet_Sorted=np.zeros((6*self.l, 19, len(self.selectedList)), dtype=float)
        selectedFeature=[]
        iter=0
        for maxList_iter in self.maxList:
            if maxList_iter in self.selectedList:
                selectedFeature.append(iter)
            iter+=1
        for iter in range(len(selectedFeature)):
            for featureIndex in range(6*self.l):
                self.divTrainSet_Sorted[featureIndex, :, iter]=self.divTrainSet[featureIndex,:,selectedFeature[iter]]
                self.divTestSet_Sorted[featureIndex, :, iter]=self.divTestSet[featureIndex,:,selectedFeature[iter]]
        print "Feature Selection Completed, Selected Features:",self.selectedList
#Break Time Series into approximately equal length l parts, l takes values from 1 to 20
#Little Note: MAX(480 % i)==12, others < 7.
    def division_op(self, target_sequence, l=1):
        avg=len(target_sequence)/float(l)
        out=[]
        last=0.0
        while last<len(target_sequence):
            out.append(target_sequence[int(last):int(last+avg)])
            last+=avg
        return out
#No postprocessData anymore, directly derive the statistical matrix
    def timeSeriesDivision(self):
        self.divTrainSet=np.zeros((6*self.l, 69, 7), dtype=float)
        self.divTestSet=np.zeros((6*self.l, 19, 7), dtype=float)
        testCnt, trainCnt= 0, 0
        for instancePnt in range(88):
            testFeaturePnt, trainFeaturePnt = 0, 0
            for tSeriesPnt in range(6):
                divOutput=self.division_op(self.raw_dataset[instancePnt, :, tSeriesPnt], self.l)
                if instancePnt+1 in self.testList:
                    for subSeries in range(self.l):
                        self.divTestSet[testFeaturePnt, testCnt, :]=self.featureExtract(divOutput[subSeries])
                        testFeaturePnt+=1
                    flag=0
                else:
                    for subSeries in range(self.l):
                        self.divTrainSet[trainFeaturePnt, trainCnt, :]=self.featureExtract(divOutput[subSeries])
                        trainFeaturePnt+=1
                    flag=1
            testCnt+=(1-flag)
            trainCnt+=flag
        print "Feature Extraction Complete, with l =",self.l
#Resize data for Logistic Regression & RFE
    def dataResize(self):
        self.testSetIn=np.zeros((19, 6 * self.l * len(self.selectedList)), dtype=float)
        self.trainSetIn=np.zeros((69, 6 * self.l * len(self.selectedList)), dtype=float)
        for testIter in range(19):
            for i in range(len(self.selectedList)-1):
                self.testSetIn[testIter][6*self.l*i:6*self.l*(i+1)]=self.divTestSet_Sorted[:, testIter, i]
            self.testSetIn[testIter][-6*self.l:]=self.divTestSet_Sorted[:, testIter, i]
        for trainIter in range(69):
            for j in range(len(self.selectedList)-1):
                self.trainSetIn[trainIter][6*self.l*j:6*self.l*(j+1)]=self.divTrainSet_Sorted[:, trainIter, j]
            self.trainSetIn[trainIter][-6*self.l:]=self.divTrainSet_Sorted[:, trainIter, j]
        print "Data ready for Logistic Regression." 
#Logistic Regression
    def logisticRegression_perform(self):
        print "Starting Logistic Regression.\n"
        logModel=LogisticRegression(max_iter=10000, C=10000, solver='lbfgs').fit(self.trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(self.trainSetIn), logModel.predict(self.testSetIn)
        probTrain, probTest = logModel.predict_proba(self.trainSetIn), logModel.predict_proba(self.testSetIn)
        np.set_printoptions(precision=3)
        print 'Rank of this 69 * 18 matrix:',np.linalg.matrix_rank(self.trainSetIn)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Pribability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        denom = (2.0*(1.0+np.cosh(logModel.decision_function(self.trainSetIn))))
        F_ij=np.zeros((18, 18), dtype=float)
        for i in range(18):
            for j in range(18):
                for ggg in range(69):
                    F_ij[i][j]+=(self.trainSetIn[ggg, i]*self.trainSetIn[ggg, j])/denom[ggg]
        F_ij2=np.dot(np.dot(self.trainSetIn.T, np.diag(denom)), self.trainSetIn)
        denom = np.tile(denom,(self.trainSetIn.shape[1],1)).T
        F_ij3 = np.dot(np.divide(self.trainSetIn, denom).T, self.trainSetIn) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij3) ## Inverse Information Matrix
        #print np.diag(Cramer_Rao)
        print 'Inverse of Fisher Information Matrix 3 is:\n',np.linalg.eigvals(Cramer_Rao)
        print 'Fisher Information Matrix 1:\n', np.linalg.eigvals(F_ij)
        print 'Fisher Information Matrix 2 is:\n', np.linalg.eigvals(F_ij2)
        print 'Fisher Information Matrix 3 is:\n', np.linalg.eigvals(F_ij3)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        #print sigma_estimates
        z_scores = logModel.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
#StatsModel Logistic Regression
    def smModel(self):
        best_Model=np.zeros((69, 13), dtype=float)
        for i in range(13):
            best_Model[:, i] = self.trainSetIn[:, self.bestfeatureList_1000[i]]
        model=sm.Logit(self.trueTrainLabel, best_Model)
        #result=model.fit()
        #print result.summary2()
#Recursive Feature Selection & Cross Validation
    def rfe_perform(self):
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        rfecvModel=LogisticRegression(max_iter=7, C=10000, solver='lbfgs')
        # The "accuracy" scoring is proportional to the number of correct classifications
        rfecv = RFECV(estimator=rfecvModel, step=1, cv=StratifiedKFold(5), scoring='accuracy')
        rfecv.fit(self.trainSetIn, self.trueTrainLabel)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylim(0.5, 1.1)
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        #plt.savefig('l=4', dpi=800)
        #plt.show()
        #Returns the best score + the numbers of feature selected + Selection of Features
        return  max(rfecv.grid_scores_), rfecv.n_features_, rfecv.support_
#Scatter Plot Matrix
    def scattorPlot(self):
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        plotFeatures=np.zeros((69, 9))
        if self.l == 1:
            plotFeatures[:, 0:3]=self.trainSetIn[:, 0:3]
            plotFeatures[:, 3:6]=self.trainSetIn[:, 3:6]
            plotFeatures[:, -3:]=self.trainSetIn[:, -3:]
        elif self.l == 2:
            plotFeatures[:, 0:3]=self.trainSetIn[:, 0:3]
            plotFeatures[:, 6:9]=self.trainSetIn[:, 6:9]
            plotFeatures[:, -3:]=self.trainSetIn[:, -3:]
        df = pd.DataFrame(plotFeatures[:, :3])
        labelList=list()
        for i in range(9):
            labelList.append("bending")
        for i in range(60):
            labelList.append("not bending")
        df['label'] = labelList
        if self.l == 1:
            df.rename(columns={0:'max_1', 1:'max_2', 2:'max_6', 3:'mean_1', 4:'mean_2', 5:'mean_6', 6:'min_1', 7:'min_2', 8:'min_6'}, inplace=True)
        elif self.l == 2:
            df.rename(columns={0:'max_1(1)', 1:'max_2(1)', 2:'max_6(2)', 3:'mean_1(1)', 4:'mean_2(1)', 5:'mean_6(2)', 6:'min_1(1)', 7:'min_2(1)', 8:'min_6(2)'}, inplace=True)
        g=sns.PairGrid(df, hue='label')
        g=g.map_diag(plt.hist, histtype="step", linewidth=1)
        g=g.map_offdiag(plt.scatter, s=5)
        if self.l == 1:
            g.savefig("scatterPlot(l=1).png", dpi=800)
        elif self.l == 2:
            g.savefig("scatterPlot(l=2).png", dpi=800)
        #plt.show()
#Best Classifier
    def bestClassifier_1000(self):
        if self.l != 19:
            return 0
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        opti_testSetIn=np.zeros((19, 13), dtype=float)
        opti_trainSetIn=np.zeros((69, 13), dtype=float)
        for iter in range(13):
            opti_testSetIn[:, iter]=self.testSetIn[:, self.bestfeatureList_1000[iter]]
            opti_trainSetIn[:, iter]=self.trainSetIn[:, self.bestfeatureList_1000[iter]]
        logModel=LogisticRegression(max_iter=1000, C=10000, solver='lbfgs').fit(opti_trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(opti_trainSetIn), logModel.predict(opti_testSetIn)
        probTrain, probTest = logModel.predict_proba(opti_trainSetIn), logModel.predict_proba(opti_testSetIn)
        np.set_printoptions(precision=5)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        print 'Distance from data points to decision hyperplane:\n', logModel.decision_function(opti_trainSetIn)
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        fpr, tpr,_=roc_curve(logModel.predict(opti_trainSetIn),self.trueTrainLabel, drop_intermediate=False)
        print roc_auc_score(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve maxiter=1000')
        plt.savefig('ROCAUC-max_iter=1000', dpi=800)
        plt.show()
        #Case-Control Sampling, introducing Bias Calibration: 0.1446693982
        #Update train Data to introduce Bias
        logModel=LogisticRegression(max_iter=1000, C=10000, solver='liblinear', fit_intercept=True, intercept_scaling=0.1446693982).fit(opti_trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(opti_trainSetIn), logModel.predict(opti_testSetIn)
        probTrain, probTest = logModel.predict_proba(opti_trainSetIn), logModel.predict_proba(opti_testSetIn)
        np.set_printoptions(precision=5)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        print 'Distance from data points to decision hyperplane:\n', logModel.decision_function(opti_trainSetIn)
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        fpr, tpr,_=roc_curve(logModel.predict(opti_trainSetIn),self.trueTrainLabel, drop_intermediate=False)
        print roc_auc_score(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve maxiter Bias=1000')
        plt.savefig('ROCAUC-max_iter=1000 Bias', dpi=800)
        plt.show()
    def bestClassifier_7(self):
        if self.l != 4:
            return 0
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        opti_testSetIn=np.zeros((19, 6), dtype=float)
        opti_trainSetIn=np.zeros((69, 6), dtype=float)
        for iter in range(6):
            opti_testSetIn[:, iter]=self.testSetIn[:, self.bestfeatureList_7[iter]]
            opti_trainSetIn[:, iter]=self.trainSetIn[:, self.bestfeatureList_7[iter]]
        logModel=LogisticRegression(max_iter=7, C=10000, solver='lbfgs').fit(opti_trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(opti_trainSetIn), logModel.predict(opti_testSetIn)
        probTrain, probTest = logModel.predict_proba(opti_trainSetIn), logModel.predict_proba(opti_testSetIn)
        np.set_printoptions(precision=5)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        print 'Distance from data points to dicision hyperplane:\n', logModel.decision_function(opti_trainSetIn)
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        fpr, tpr,_=roc_curve(logModel.predict(opti_trainSetIn),self.trueTrainLabel, drop_intermediate=False)
        print roc_auc_score(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve maxiter=7')
        plt.savefig('ROCAUC-max_iter=7', dpi=800)
        plt.show()
        #Case-Control Sampling, introducing Bias Calibration: 0.1446693982
        #Update train Data to introduce Bias
        logModel=LogisticRegression(max_iter=7, C=10000, solver='liblinear', fit_intercept=True, intercept_scaling=0.1446693982).fit(opti_trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(opti_trainSetIn), logModel.predict(opti_testSetIn)
        probTrain, probTest = logModel.predict_proba(opti_trainSetIn), logModel.predict_proba(opti_testSetIn)
        np.set_printoptions(precision=5)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        print 'Distance from data points to dicision hyperplane:\n', logModel.decision_function(opti_trainSetIn)
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        fpr, tpr,_=roc_curve(logModel.predict(opti_trainSetIn),self.trueTrainLabel, drop_intermediate=False)
        print roc_auc_score(logModel.predict(opti_trainSetIn),self.trueTrainLabel)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve maxiter Bias=7')
        plt.savefig('ROCAUC-max_iter=7 Bias', dpi=800)
        plt.show()
#L1 Penalized Logistic Regression
    def l1_perform(self):
        regStrength=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        regStrength=np.exp(regStrength)
        lassoModel=LogisticRegressionCV(Cs=regStrength, penalty='l1', solver='liblinear', cv=5, refit=True)
        lassoModel.fit(self.trainSetIn, self.trueTrainLabel)
        #Returns the best score + Best C
        return  max(lassoModel.scores_), lassoModel.C_
#Best Classifier with L1 Penalty
    def bestClassifierL1(self):
        if self.l != self.bestl1:
            return 0
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        logModel=LogisticRegression(max_iter=10000, C=self.bestAlpha, solver='liblinear', penalty='l1')
        logModel.fit(self.trainSetIn, self.trueTrainLabel)
        resultTrain, resultTest = logModel.predict(self.trainSetIn), logModel.predict(self.testSetIn)
        probTrain, probTest = logModel.predict_proba(self.trainSetIn), logModel.predict_proba(self.testSetIn)
        np.set_printoptions(precision=5)
        print 'The Train Result:\n', resultTrain
        print 'The Test Result:\n', resultTest
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        print 'Coefficient for Logistic Regression:\n', logModel.coef_
        print 'Distance from data points to dicision hyperplane:\n', logModel.decision_function(self.trainSetIn)
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(logModel.predict(self.trainSetIn),self.trueTrainLabel)
        fpr, tpr,_=roc_curve(logModel.predict(self.trainSetIn),self.trueTrainLabel, drop_intermediate=False)
        print roc_auc_score(logModel.predict(self.trainSetIn),self.trueTrainLabel)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve L1')
        plt.savefig('ROCAUC L1', dpi=800)
        plt.show()
        #RFE Comparison
        rfeModel=LogisticRegression()
        rfe=RFE(rfeModel, 8)
        rfe=rfe.fit(self.trainSetIn, self.trueTrainLabel)
        print 'RFE Feature Selection Results:\n', rfe.support_
        print 'RFE Feature Ranking:\n', rfe.ranking_
#MultiNomial Classification
    def multiToyTest(self):
        multiModel=LogisticRegression(max_iter=10000, penalty='l1', solver='saga', multi_class='multinomial')
        multiModel.fit(self.trainSetIn, self.trueMultiTrainLabel)
        print 'Result for Direct Fit:'
        print 'Train Result:\n', multiModel.predict(self.trainSetIn)
        print 'True Train Label:\n', self.trueMultiTrainLabel
        print 'Test Result:\n', multiModel.predict(self.testSetIn)
        print 'True Test Label:\n', self.trueMultiTestLabel
    def multi_perform(self):
        regStrength=np.exp([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        multiModel=LogisticRegressionCV(max_iter=10000, Cs=regStrength, penalty='l1', solver='saga', cv=5, refit=True, multi_class='multinomial')
        multiModel.fit(self.trainSetIn, self.trueMultiTrainLabel)
        print 'Cross Validation Complete'
        #Refit the Model with Optimal C_
        opti_strength=[np.mean(multiModel.C_)]
        multiModel.set_params(Cs=opti_strength)
        multiModel.fit(self.trainSetIn, self.trueMultiTrainLabel)
        scores=np.zeros((6), dtype=float)
        for i in range(6):
            scores[i]=np.mean(multiModel.scores_[i])
        #Returns the best score array + Best C
        return  scores, multiModel.C_
    def bestClassifier_multi(self):
        if self.l != 5:
            return 0
        multiModel=LogisticRegressionCV(max_iter=10000, Cs=[self.bestMultiAlpha], penalty='l1', solver='saga', cv=5, refit=True, multi_class='multinomial')
        multiModel.fit(self.trainSetIn, self.trueMultiTrainLabel)
        resultTrain, resultTest = multiModel.predict(self.trainSetIn), multiModel.predict(self.testSetIn)
        probTrain, probTest = multiModel.predict_proba(self.trainSetIn), multiModel.predict_proba(self.testSetIn)
        np.set_printoptions(precision=3)
        print 'The Train Result:\n', resultTrain
        print 'True Label of train Set:\n', self.trueMultiTrainLabel
        print 'The Test Result:\n', resultTest
        print 'True Label of test Set:\n', self.trueMultiTestLabel
        print 'Probability estimated for TrainSet:\n', probTrain
        print 'Probability estimated for TestSet:\n', probTest
        #DRAW ROC & AOC & Confusion Matrix
        ##Computing false and true positive rates
        print confusion_matrix(multiModel.predict(self.trainSetIn),self.trueMultiTrainLabel)
        print confusion_matrix(multiModel.predict(self.testSetIn), self.trueMultiTestLabel)
        self.drawRocAucCurve()
    def drawRocAucCurve(self):
        os.chdir("/Users/Gaara/Desktop/USC/EE559/Homework/Homework34/")
        classList=list(['bending', 'cycling', 'lying', 'sitting', 'standing', 'walking'])
        X_train = self.trainSetIn
        X_test = self.testSetIn
        y_train = label_binarize(self.trueMultiTrainLabel, classes=[0, 1, 2, 3, 4, 5])
        y_test = label_binarize(self.trueMultiTestLabel, classes=[0, 1, 2, 3, 4, 5])
        multiModel = OneVsRestClassifier(LogisticRegression(penalty='l1', C=self.bestMultiAlpha, max_iter=10000, solver='liblinear', multi_class='ovr'))
        y_score = multiModel.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, roc_auc= dict(), dict(), dict()
        for i in range(6):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(6):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 6
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(6), colors):
            plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'''.format(classList[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig("MultiClass ROC-AUC Curve", dpi=800)
        plt.show()
#Naive Bayes Classification
    def multiNomialBayes(self):
        np.set_printoptions(precision=4)
        print 'Result based on MultiNomial Prior:'
        guassianModel=MultinomialNB().fit(self.trainSetIn, self.trueMultiTrainLabel)
        print 'True Train Label:\n', self.trueMultiTrainLabel
        print 'Predicted Train Label:\n', guassianModel.predict(self.trainSetIn)
        print 'True Test Label:\n', self.trueMultiTestLabel
        print 'Predicted Test Label:\n', guassianModel.predict(self.testSetIn)
        print 'Predicted Train Probability:\n', guassianModel.predict_proba(self.trainSetIn)
        print 'Predicted Test Probability:\n',  guassianModel.predict_proba(self.testSetIn)
        print 'Predict Accuracy on Train Set:\n', guassianModel.score(self.trainSetIn, self.trueMultiTrainLabel)
        print 'Predict Accuracy on Test Set:\n', guassianModel.score(self.testSetIn, self.trueMultiTestLabel)
    def gaussianBayes(self):
        np.set_printoptions(precision=4)
        print 'Result based on Gaussian Prior:'
        guassianModel=GaussianNB().fit(self.trainSetIn, self.trueMultiTrainLabel)
        print 'True Train Label:\n', self.trueMultiTrainLabel
        print 'Predicted Train Label:\n', guassianModel.predict(self.trainSetIn)
        print 'True Test Label:\n', self.trueMultiTestLabel
        print 'Predicted Test Label:\n', guassianModel.predict(self.testSetIn)
        print 'Predicted Train Probability:\n', guassianModel.predict_proba(self.trainSetIn)
        print 'Predicted Test Probability:\n',  guassianModel.predict_proba(self.testSetIn)
        print 'Predict Accuracy on Train Set:\n', guassianModel.score(self.trainSetIn, self.trueMultiTrainLabel)
        print 'Predict Accuracy on Test Set:\n', guassianModel.score(self.testSetIn, self.trueMultiTestLabel)
def timeDivision_RFECV():
    bestScore=0
    bestNumFeature=1000
    for fold in range(1, 21):
        rua=hw3(fold)
        best_temp, bestNumFeature_temp, featureSet = rua.rfe_perform()
        if (best_temp>bestScore)|((best_temp==bestScore)&(bestNumFeature_temp < bestNumFeature)):
            bestScore = best_temp
            bestNumFeature = bestNumFeature_temp
            bestFeatureSet = featureSet
            bestFold = fold
    print "Best Correct Rate: ", bestScore
    print "Number of Features: ", bestNumFeature
    print "Selected Feature Set: ", bestFeatureSet
    print "Fold which optimal circumstances is in: ", bestFold
def timeDivision_l1LassoCV():
    bestScore=0
    for fold in range(1, 21):
        rua=hw3(fold)
        best_temp, alpha = rua.l1_perform()
        if (best_temp>bestScore):
            bestScore = best_temp
            bestFold = fold
            bestAlpha = alpha
    print "Best Correct Rate: ", bestScore
    print "Fold which optimal circumstances is in: ", bestFold
    print "Optimal alpha: ", bestAlpha
    rua=hw3(bestFold)
def timeDivision_multiCV():
    bestScore=0
    prior=[0.1477, 0.1705, 0.1705, 0.1705, 0.1705, 0.1705]
    for fold in range(1, 21):
        rua=hw3(fold)
        score_array, alpha = rua.multi_perform()
        print np.dot(prior, score_array)
        if ( np.dot(prior, score_array) > bestScore):
            best_scorrArray = score_array
            bestScore = np.dot(prior, score_array)
            bestFold = fold
            bestAlpha = np.mean(alpha)
    print "Optimal Score of each class:\n", best_scorrArray
    print 'Average score with prior knowledge:\n', np.dot(prior, best_scorrArray)
    print "Fold which optimal circumstances is in: ", bestFold
    print "Optimal alpha: ", bestAlpha
#timeDivision_RFECV()
#testModel=hw3(1)
#testModel.rfe_perform()
#testModel.scattorPlot()
#testModel.logisticRegression_perform()
#testModel.smModel()
#testModel.multiToyTest()

#BestModel_1000=hw3(4)
#BestModel_1000.bestClassifier_1000()
#BestModel_7=hw3(19)
#BestModel_7.bestClassifier_7()
#BestModel_7.smModel()

#timeDivision_l1LassoCV()
#BestModel_l1=hw3(1)
#BestModel_l1.bestClassifierL1()

#timeDivision_multiCV()
#BestModel_multi=hw3(5)
#BestModel_multi.bestClassifier_multi()
#BestModel_multi.drawRocAucCurve()

bayesModel=hw3(5)
bayesModel.multiNomialBayes()
bayesModel.gaussianBayes()
