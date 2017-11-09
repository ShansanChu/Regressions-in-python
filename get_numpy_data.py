""" 
   this module is to preparetion of regression in sci-learn as the input should be arrays rather sframe
"""
import pandas
import numpy as np
def get_numpy_data(df,y):
     (length,num_c)=df.shape
     #print df.shape
     ones_arr=np.ones((length,1),dtype=float)
     try:
       features_arr=df.drop(y,1).astype(float).values
       features_arr=np.concatenate((ones_arr,features_arr),axis=1)
       #features_arr= map(lambda x: float(x),features_arr)
       output_arr=df[y].values.reshape((length,1))
       #print output_arr.shape
       return features_arr,output_arr
     except:
       print 'Eooroors with the columns name'
       return False
