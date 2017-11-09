"""
this module is to do the assignents for week 6. To query the K nearest neighbors for small data set
"""
import pandas
import numpy as np
from numpy import linalg as LA
from math import sqrt as sqrt
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
#sales=pandas.read_csv('kc_house_data_small.csv')
sales=pandas.read_csv('kc_house_data_small.csv')
train_data=pandas.read_csv('kc_house_data_small_train.csv')
test_data=pandas.read_csv('kc_house_data_small_test.csv')
vali_data=pandas.read_csv('kc_house_data_validation.csv')
print len(train_data),len(test_data),len(vali_data),len(sales)
def get_numpy_data(df,features,y):
     (length,num_c)=df.shape
     ones_arr=np.ones((length,1),dtype=float)
     try:
       features_arr=df[features].astype(float).values
       features_arr=np.concatenate((ones_arr,features_arr),axis=1)
       #features_arr= map(lambda x: float(x),features_arr)
       output_arr=df[y].values
       return features_arr,output_arr
     except:
       print 'Eooroors with the columns name'
       return False

def normalize_features(features):
    """
    input is feature array. Employing numpy to do normalization.
    one column is corresponding to one feature.
    """
    norms=LA.norm(features,axis=0)
    NR_features=features/norms
    return NR_features,norms

features=['bathrooms', 'waterfront','sqft_above','sqft_living15', 'grade', 'yr_renovated', 'bedrooms', 'zipcode', 'long','sqft_lot15', 'sqft_living', 'floors', 'condition', 'lat', 'sqft_basement', 'yr_built', 'sqft_lot', 'view']
output='price'
#small_fea,small_output=get_numpy_data(sales,features,output)
train_fea,train_output=get_numpy_data(train_data,features,output)
test_fea,test_output=get_numpy_data(test_data,features,output)
valid_fea,valid_output=get_numpy_data(vali_data,features,output)
train_fea, norms = normalize_features(train_fea)
print norms
test_fea =test_fea/norms
valid_fea =valid_fea/norms
print test_fea[0]
print train_fea[9]
dis=sqrt(np.sum((test_fea[0]-train_fea[9])**2))
print dis
n=10
dis_arr=np.zeros(n)
dis_min=1.0e12
for i in range(10):
      dis_arr[i]=sqrt(np.sum(test_fea[0]-train_fea[i])**2)
      if dis_min>dis_arr[i]:
           dis_min=dis_arr[i]
print dis_arr
print dis_min
diff=train_fea-test_fea[0]
nn=len(diff)
print diff[-1].sum(),train_fea[-1],test_fea[0]
distances=np.sqrt(np.sum(diff**2, axis=1))
print distances[100]
def compute_distances(features_instances, feature_query):
    diff=(features_instances-feature_query)
    distances=np.sqrt(np.sum(diff**2,axis=1))
    return distances
distances_3=compute_distances(train_fea,test_fea[2])
minimum=np.argmin(distances_3)
#index=np.where(distances_3==minimum)
print minimum,train_output[minimum]

def k_nearest_neighbors(k, feature_train, feature_query):
    diff=(feature_train-feature_query)
    distances=np.sqrt(np.sum(diff**2,axis=1))
    sorting=np.argsort(distances)
    neighbors=sorting[:k]
    return neighbors
neigh_k=k_nearest_neighbors(4,train_fea,test_fea[2])
print neigh_k
def predict_output_of_query(k, feature_train, output_train, feature_query):
    #for one query
    diff=(feature_train-feature_query)
    distances=np.sqrt(np.sum(diff**2,axis=1))
    sorting=np.argsort(distances)
    neighbors=sorting[:k]
    prediction=output_train[neighbors].sum()/k
    return prediction
price_k=predict_output_of_query(4,train_fea,train_output,test_fea[2])
print price_k
def predict_output(k, feature_train, output_train, feature_query):
    #for multiple query
    n=len(feature_query)
    predictions=np.zeros(n)
    for i in range(n):
         diff=(feature_train-feature_query[i])
         distances=np.sqrt(np.sum(diff**2,axis=1))
         sorting=np.argsort(distances)
         neighbors=sorting[:k]
         predictions[i]=output_train[neighbors].sum()/k
    return predictions
prices=predict_output(10,train_fea,train_output,test_fea[:10])
print prices,'$$$'
RSS=[]
for k in range(15):
    predictions=predict_output(k+1,train_fea,train_output,valid_fea)
    RSS.append(np.sum((valid_output-predictions)**2))
print RSS
print min(RSS)
prediction_test=predict_output(8,train_fea,train_output,test_fea)
print np.sum((prediction_test-test_output)**2)
