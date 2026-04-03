import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('WinQT.csv')
#print(df.head(10))
#--------------classes of target values------------
#print(set(df.quality))
# description of the dataset
# df.info()
# df.describe()
# print(df.isnull().sum())
#df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
#figuring out the correlation
plt.figure(figsize=(12, 12))
corr=df.corr()
sb.heatmap(corr,annot=True)
#plt.show()
#detect  the outliers
z=np.abs(stats.zscore(df))
#print(z)
# z>3 this is outliers
print(np.where(z>3))
print(df.shape)
#remove those values with z>3
new_data=df[(z<3).all(axis=1)]
#print(new_data.shape)
features = new_data.drop(['quality', 'Id'],axis=1)
target = new_data['quality']
xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2)
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
rf_class_model=RandomForestClassifier(n_estimators=200)
rf_class_model.fit(xtrain,ytrain)
y_pred=rf_class_model.predict(xtest)
print(y_pred)
#data evaluation
print('Accuracy Score',metrics.accuracy_score(ytest,y_pred))
#tree vis
# plt.figure(figsize=(20, 20))
# tree.plot_tree(rf_class_model.estimators_[0],filled=True)
# plt.show()
# plt.figure(figsize=(20, 20))
# for i in range(3):  # only first 3 trees
#     plt.figure(figsize=(20, 20))
#     tree.plot_tree(rf_class_model.estimators_[i], filled=True)
#     plt.title(f"Tree {i+1}")
#     plt.show()
import pickle

# Save model
with open('wine_model.pkl', 'wb') as f:
    pickle.dump(rf_class_model, f)
# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)