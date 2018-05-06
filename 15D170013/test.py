
from train import get_features_from_input

import pandas as pd

# Included the df_train as well since in the get_features_from_input function while doing the 
# one hot encoding we need all the possible features

df_test = pd.read_csv('testSold.csv', index_col=0)
df_train = pd.read_csv('trainSold.csv', index_col=0)
df_train.drop('SaleStatus', axis=1, inplace=True)

df_gt = pd.read_csv('gt.csv', index_col=0)

y = df_gt.as_matrix()

X = get_features_from_input(df_test, df_train)

#----------------------------------------------------------------------------------------

#Read the models in as input from the model files

import pickle

best_model1 = pickle.load(open('finalModel1.pkl', 'rb'))

best_model2 = pickle.load(open('finalModel2.pkl', 'rb'))

#----------------------------------------------------------------------------------------


#Calculating the scores for both the models
sc1 = best_model1.score(X, y)
sc2 = best_model2.score(X, y)

print("Accuracy of model1 is ", sc1)
print("Accuracy of model2 is ", sc2)

#----------------------------------------------------------------------------------------

#Predict the output for the input data for the better model and saving the output to out.csv
out = []
if (sc1 > sc2):
    out = best_model1.predict(X)
else:
    out = best_model2.predict(X)

out_df = pd.DataFrame({'Id' : df_test.index.values, 'SaleStatus' : out[:]})
out_df.to_csv('out.csv', index=False)