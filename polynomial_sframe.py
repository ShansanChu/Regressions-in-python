""" 
    this module is part of funactions used by assigment
    this module is also used in week3's assigment.
"""
import pandas
def add_powers(dataset,column_name,new_name,power):
     """
     !!!!!!!!!
     #input dataset is a pandas dataframe
     #column_name is the column to be as x as power function
     #new_name is a string which is the name of new columns to append to dataset
     #power is integer which will be needed to do x**power performance.
     !!!!!!!!!
     #output is the dataset after adding a new frame
     """
     dataset[new_name]=dataset[column_name].apply(lambda x: x**power)
     return dataset

def polynomial_sframe(feature,degree):
    """ 
    return a new dataframe with every column corresponding to power of features_coloumns
    !!!
    input is array name features and degree is the maximal power of these features
    """
    #create an enpty DataFrame
    poly_dataF=pandas.DataFrame()
    poly_dataF['power_1']=feature
    if degree>1:
        for power in range(2,degree+1):
           name='power_'+str(power)
           poly_dataF[name]=poly_dataF['power_1'].apply(lambda x: x**power)
    return poly_dataF
