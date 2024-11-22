import pandas as pandamodule
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class ModelData:
    featuresEnc = LabelEncoder()
    labelEnc = LabelEncoder()
    def __init__ (self):
        pass
    
    def loadCSV(self): 
        return pandamodule.read_csv('co2_emissions_data.csv')

    def checkMissingValues(self,data):
        return data.isnull().sum()
    
    def checkScale(self,data):
        return data.describe()
    
    def showPairPlot(self,data):
        sns.pairplot(data, diag_kind='hist')
        plt.show()

    def showHeatMap(self,data):
        numeric_data = data.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

    def seperateTargets(self,data,targets):
        X = data.drop(targets, axis=1)
        Y1 = data[targets[0]]
        Y2 = data[targets[1]]
        return X,Y1,Y2


    def convertNumerical(self,data,catColumns):
        numericalData = copy.deepcopy(data)
        y = copy.deepcopy(data["Emission Class"])
        for cat in catColumns:
            self.featuresEnc.fit(data[cat].unique())
            numericalData[cat] = self.featuresEnc.transform(data[cat])
        self.labelEnc.fit(y)
        self.labelEnc.transform(y)
        return numericalData
    
    def inverseYPredict(self,value):
        return self.labelEnc.inverse_transform(value)
    
    def split(self, X, Y1,Y2, testSize):
        xTrain, xTest, y1Train, y1Test, y2Train, y2Test = train_test_split(X, Y1, Y2, test_size=testSize, random_state=42)
        return xTrain,xTest,y1Train,y1Test,y2Train,y2Test
    
    def scale(self,data):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data)
        return pandamodule.DataFrame(scaled_features, columns=data.columns)

