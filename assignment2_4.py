"""
   this module is to do the second assignment which implementing the Ridge regression.
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
#dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales=pandas.read_csv('kc_house_data1.csv')
test_data=pandas.read_csv('kc_house_test_data.csv')
train_data=pandas.read_csv('kc_house_train_data.csv')
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
def predict_outcome(features_arr,weights):
     return np.dot(features_arr,weights)
def feature_derivative_ridge(errors, feature,l2_pl,omega,feature_is_cs):
     if feature_is_cs:
         derivative=2.0*np.dot(feature,errors)
     else:
         derivative=2.0*np.dot(feature,errors)+2*l2_pl*omega
     return derivative

def regression_gradient_ridge(features_arr, output_arr, initial_weights, step_size, l2_pl,max_it=100):
     weights=np.array(initial_weights)
     loop=0
     while True:
        predicted=predict_outcome(features_arr,weights)
        errors=predicted-output_arr
        for i in range(len(weights)):
            is_cs=False
            if i==0:
                is_cs=True
            derivative=feature_derivative_ridge(errors,features_arr[:,i],l2_pl,weights[i],is_cs)
            weights[i]=weights[i]-step_size*derivative
            loop=loop+1
        if loop>max_it:
            break
     return weights
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
print len(output),len(simple_feature_matrix)
step_size = 1e-12
l2_penalty=1e11
max_iterations = 1000
initial_weights = np.array([0.0, 0.0])
simple_weights_0_penalty=regression_gradient_ridge(simple_feature_matrix,output,initial_weights,step_size,0.0,max_iterations)
print simple_weights_0_penalty
simple_weights_high_penalty=regression_gradient_ridge(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print simple_weights_high_penalty
RSS1=np.sum((predict_outcome(simple_test_feature_matrix, simple_weights_0_penalty)-test_output)**2)
RSS2=np.sum((predict_outcome(simple_test_feature_matrix, simple_weights_high_penalty)-test_output)**2)
RSS3=np.sum((predict_outcome(simple_test_feature_matrix, initial_weights)-test_output)**2)
print RSS1,RSS2,RSS3
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
initial_weights1 = np.array([0.0, 0.0, 0.0])
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output_1) = get_numpy_data(test_data, model_features, my_output)
multiple_weights_0_penalty=regression_gradient_ridge(feature_matrix,output,initial_weights1,step_size,0.0,max_iterations)
multiple_weights_high_penalty=regression_gradient_ridge(feature_matrix,output,initial_weights1,step_size,l2_penalty,max_iterations)
print multiple_weights_0_penalty
print multiple_weights_high_penalty
RSS_1=np.sum((predict_outcome(test_feature_matrix, multiple_weights_0_penalty)-test_output_1)**2)
RSS_2=np.sum((predict_outcome(test_feature_matrix, multiple_weights_high_penalty)-test_output_1)**2)
RSS_3=np.sum((predict_outcome(test_feature_matrix, initial_weights1)-test_output_1)**2)
print RSS_1,RSS_2,RSS_3
pred_0=predict_outcome(test_feature_matrix, multiple_weights_0_penalty)
pred_high=predict_outcome(test_feature_matrix, multiple_weights_high_penalty)
print test_output_1[0],pred_0[0],pred_high[0]
plt.plot(simple_feature_matrix,output,'k.',simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_0_penalty),'b-',simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_high_penalty),'r-')
plt.show()
