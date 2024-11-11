import pandas as pandamodule
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

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
    

DA = ModelData()
# Load the "co2_emissions_data.csv" dataset
data = DA.loadCSV()
# check whether there are missing values
missingValues = DA.checkMissingValues(data)
# print(missingValues)
# check whether numeric features have the same scale
dataScaleChecking = DA.checkScale(data)
# print(dataScaleChecking)
# visualize a pairplot in which diagonal subplots are histograms
# DA.showPairPlot(data)
# visualize a correlation heatmap between numeric columns
# DA.showHeatMap(data)
# categorical features and targets are encoded
numericalData = DA.convertNumerical(data,["Make","Model","Vehicle Class","Transmission","Fuel Type","Emission Class"])
# print(numericalData)
# the features and targets are separated
X,Y1,Y2 = DA.seperateTargets(numericalData,["CO2 Emissions(g/km)","Emission Class"])
# print("X = ", X)
# print("Y1 = ", Y1)
# print("Y2 = ", Y2)
# numeric features are scaled
X = DA.scale(X)
# print(X)
# the data is shuffled and split into training and testing sets
xTrain,xTest,y1Train,y1Test,y2Train,y2Test = DA.split(X,Y1,Y2,0.2)
# print("xTrain = ", xTrain)
# print("xTest = ", xTest)
# print("y1Train = ", y1Train)
# print("y1Test = ", y1Test)
# print("y2Train = ", y2Train)
# print("y2Test = ", y2Test)