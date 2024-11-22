import numpy
import numpy as np
import pandas as pd

from preprocessing_and_analysis import ModelData
from multi_features_gradient_descent import LinearRegression
from LogisticRegression import LogisticRegressionModel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, confusion_matrix
import seaborn as sns


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


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

# the data is shuffled and split into training and testing sets
xTrain,xTest,y1Train,y1Test,y2Train,y2Test = DA.split(X,Y1,Y2,0.3)
# print("xTrain = ", xTrain)
# print("xTest = ", xTest)
# print("y1Train = ", y1Train)
# print("y1Test = ", y1Test)
# print("y2Train = ", y2Train)
# print("y2Test = ", y2Test)
# DA.showHeatMap(numericalData)


# numeric features are scaled
X = DA.scale(X)
# print(X)





x_train = xTrain[['Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']].to_numpy()
y_train = y1Train.to_numpy()

x_test = xTest[['Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']].to_numpy()
y_test = y1Test.to_numpy()

#Linear Regression

# obj = LinearRegression(x_train,y_train, 0.01, 400)
# costs = obj.run_linear_regression()
#
#
# y_predict_train = obj.predict(x_train)
# R2_sklearn_train = r2_score(y_train, y_predict_train)
# print(f"R² Train Data (Scikit-Learn Calculation): {R2_sklearn_train}")
#
#
# y_predict_test = obj.predict(x_test)
# R2_sklearn = r2_score(y_test, y_predict_test)
# print(f"R² test Data (Scikit-Learn Calculation): {R2_sklearn}")



# mape = mean_absolute_percentage_error(y_test, y_predict)
# print(f"MAPE: {mape * 100:.2f}%")

# plt.plot(costs,linewidth=4)
# plt.title('Error Reduction During Gradient Descent')
# plt.xlabel('Iterations')
# plt.ylabel('Cost/Error')
# plt.grid()
# plt.show()


#Logistic Regression
y2_train = y2Train.to_numpy()
y2_test = y2Test.to_numpy()


log_obj = LogisticRegressionModel(x_train,y2_train,x_test,y2_test)
y_pred = log_obj.run_log()
accuracy = log_obj.calc_accuracy(y_pred)
print(f'Accuracy: {accuracy*100:.2f}')


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y2_test, y_pred, labels = [0,1,2,3]), annot=True, fmt='d', cmap='Blues',
            xticklabels=['HIGH', 'LOW', 'MODERATE', 'VERY LOW'], yticklabels=['HIGH', 'LOW', 'MODERATE', 'VERY LOW'])

plt.xlabel('Predicted' ,labelpad=20)
plt.ylabel('True Label', labelpad=20)
plt.title('Confusion Matrix')
plt.show()


