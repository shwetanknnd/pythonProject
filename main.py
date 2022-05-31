import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sb

Dataframe = pd.read_csv(R"/Users/shwetankanand/Downloads/winequality.csv")  # Load the data which is downloaded from keegal.com and stoed in path
Dataframe.head()                                                            # disply the total number of coloumns coloumns

Dataframe.info()                                                            # informatin of the data base values
Dataframe.isnull().sum()                                                    # Null fileds in the database

correlations = Dataframe.corr()["quality"].drop("quality")                  # find the corelation of each varible with quality of wine corr() is to find corelation between entity
# the corelation cofficient with quality is highest and +ve for alchol which means quality is
# higlely dependednt on the alchol volume in the wine
print(correlations)

sb.heatmap(Dataframe.corr())
plt.show()

# function define feature where it takes those feature whose corealation is above the threasshold value i.e. .05
col = []

for i in range(len(Dataframe.corr().keys())):                               # loop for columns
    j = 0
    for j in range(j):                                                      # loop for row
        if abs(Dataframe.corr().iloc[i,j]) > 0.05:
            col = Dataframe.corr().columns[i]
            print(col)

new_df = Dataframe                                                          #assigne new data frme to new df
new_df.update(new_df.fillna(new_df.mean()))
print("mean value of new data frame ",new_df.mean())

cat = new_df.select_dtypes(include='O')

df_dummies = pd.get_dummies(new_df,drop_first = True)                       #create dummy sets of coloumns
print(df_dummies)


df_dummies['best quality']=[1 if x>=7 else 0 for x in Dataframe.quality]
print(df_dummies)

x = df_dummies.drop(['quality','best quality'],axis=1)                      #Indipendent varibles
y = df_dummies['best quality']                                              #dependent varibles
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3) # split data to train and test model

regression = LinearRegression()                                             #assigne Linear regression to regression
regression.fit(x_train, y_train)                                            #train regressin fit for x and y training set

print("regression cofficient is ", regression.coef_)

train_pred = regression.predict(x_train)                                    #predict values for x
print("train for prediction of data sset",train_pred)
test_pred = regression.predict(x_test)                                      #predict values for y
print("testign of regression data set",test_pred)

train_rmse = mean_squared_error(train_pred, y_train) ** 0.5                 # calculate root mean square using skeliton matric mean square error
print("training root mena square value is ",train_rmse)
test_rmse = mean_squared_error(test_pred, y_test) ** 0.5                    #calulate root mean square error for the test data matric usinf skeliton matric mse
print("testing root mean square value is ",test_rmse)


predicted_data = np.round_(test_pred)
print(predicted_data)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

coeffecients = pd.DataFrame(regression.coef_)                               # regression cofficient for the values in the data.
coeffecients.columns = ['Coeffecient']
print("associcated regression cofficient", coeffecients)
