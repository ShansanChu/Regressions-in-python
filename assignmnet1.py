""" 
   this module is to do the assigmnet 1 for week 4
"""
from polynomial_sframe import polynomial_sframe as polynomial_sframe
from get_numpy_data import get_numpy_data as get_numpy_data
import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as linear_model
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pandas.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])
poly15_data = polynomial_sframe(sales['sqft_living'], 15) # use equivalent of `polynomial_sframe`
poly15_data['price']=sales['price']
my_output= 'price'
l2_small_penalty = 1.5e-5
(fit_x,fit_y)=get_numpy_data(poly15_data,my_output)
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(fit_x,fit_y)
print model.coef_
dataset1=pandas.read_csv('wk3_kc_house_set_1_data.csv')
dataset2=pandas.read_csv('wk3_kc_house_set_2_data.csv')
dataset3=pandas.read_csv('wk3_kc_house_set_3_data.csv')
dataset4=pandas.read_csv('wk3_kc_house_set_4_data.csv')
poly15_data=polynomial_sframe(dataset1['sqft_living'],15)
poly15_data['price']=dataset1['price']
poly15_data_2=polynomial_sframe(dataset2['sqft_living'],15)
poly15_data_2['price']=dataset2['price']
poly15_data_3=polynomial_sframe(dataset3['sqft_living'],15)
poly15_data_3['price']=dataset3['price']
poly15_data_4=polynomial_sframe(dataset4['sqft_living'],15)
poly15_data_4['price']=dataset4['price']
(fit15_x,fit15_y)=get_numpy_data(poly15_data,my_output)
(fit15_x_2,fit15_y_2)=get_numpy_data(poly15_data_2,my_output)
(fit15_x_3,fit15_y_3)=get_numpy_data(poly15_data_3,my_output)
(fit15_x_4,fit15_y_4)=get_numpy_data(poly15_data_4,my_output)
"""
   test difference between different lambda regulation parameter
"""
l2=[1e-9,1.23e2]
for l2_small_penalty in l2:
    model1 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model2 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model3 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model4 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model1.fit(fit15_x,fit15_y)
    model2.fit(fit15_x_2,fit15_y_2)
    model3.fit(fit15_x_3,fit15_y_3)
    model4.fit(fit15_x_4,fit15_y_4)
    plt.figure()
    plt.plot(dataset1['sqft_living'],fit15_y,'.',dataset1['sqft_living'],model1.predict(fit15_x),'-',dataset2['sqft_living'],model2.predict(fit15_x_2),'r-',dataset3['sqft_living'],model3.predict(fit15_x_3),'g-',dataset4['sqft_living'],model4.predict(fit15_x_4),'b-')
    print model1.coef_
    print model2.coef_
    print model3.coef_
    print model4.coef_

""" 
   employing cross-validation
"""
train_valid_shuffled = pandas.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pandas.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
#n=len(train_valid_shuffled)
k=10
def k_fold_cross_validation(k, l, data, output):
    n=len(data)
    error=0.0
    model_k=linear_model.Ridge(alpha=l, normalize=True)
    for i in range(k):
        start=(n*i)/k
        end=(n*(i+1))/k
        print start,'------',end
        test_set=data[start:end+1]
        test_set_y=output[start:end+1]
        train_set=data[0:start].append(data[end+1:n])
        train_set_y=output[0:start].append(output[end+1:n])
        model_k.fit(train_set,train_set_y)
        #print model_k.coef_
        error0=np.sum((model_k.predict(test_set)-test_set_y)**2)
        print error0
        error=error+error0
    return error/k
train_valid_s15=polynomial_sframe(train_valid_shuffled['sqft_living'],15) 
test_15=polynomial_sframe(test['sqft_living'],15) 
err=[]
for i in np.logspace(2, 8, num=13):
    print '****',i
    err0=k_fold_cross_validation(k,i,train_valid_s15,train_valid_shuffled['price'])
    print err0
    err.append(err0)
print err
print min(err)
lmin=100.0
model_k=linear_model.Ridge(alpha=lmin, normalize=True)
model_k.fit(train_valid_s15,train_valid_shuffled['price'])
RSS=np.sum((model_k.predict(test_15)-test['price'])**2)
#print len(test_15),len((model_k.predict(test_15)-test['price'])**2)
print RSS
plt.plot(err)
plt.show()
