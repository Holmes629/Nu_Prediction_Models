print('Initializing the job..... Setting up the workspace...')

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import math


print('Loading the datasets...')
data = pd.read_csv("extracted_data_csv_3.csv")

print('Processing the data....')

Pr= data["Pr "]
Ra= data['Ra']
Nu_train= data['Nu']
Aspect_ratio= data['Aspect_ratio']
Shape= data['Shape']
Boundary_cond= data['Boundary_conditions']
All_combined_train = pd.DataFrame({'Pr ': Pr, 'Ra': Ra, 'Aspect_ratio': Aspect_ratio, 'Shape': Shape, 'Boundary_conditions': Boundary_cond})
Nu_train= [int(i) for i in Nu_train]

########### -- Applying Logarithmic Transformation #########

Pr_train= [math.log(float(i)) for i in All_combined_train['Pr ']]
Ra_train= [math.log(float(i)) for i in All_combined_train['Ra']]
Aspect_ratio_train= All_combined_train['Aspect_ratio']
Shape_train= All_combined_train['Shape']
Boundary_conditions_train= All_combined_train['Boundary_conditions']
All_combined_train_log= pd.DataFrame({'Pr ': Pr_train, 'Ra': Ra_train, 'Aspect_ratio': Aspect_ratio_train, 'Shape': Shape_train, 'Boundary_conditions': Boundary_conditions_train})

print('Loading the models.....')

models= {
    '1 Linear Regression': LinearRegression(),
    '1 SVM': SVC(kernel = 'poly', degree = 2, C = 100000),
    '1 Random Forest': RandomForestClassifier(),
    '2 Log Linear Regression': LinearRegression(),
    '2 Log SVM': SVC(kernel = 'poly', degree = 2, C = 100000),
    '2 Log Random Forest': RandomForestClassifier(),
}

def Train_Models():
    for name, model in models.items():
        print(f'Training of {name} model:')
        if (name[0]=='1'):  
            model.fit(All_combined_train, Nu_train)
        else:
            model.fit(All_combined_train_log, Nu_train)
        print(f'    Training of {name} completed....')
print('Starting the training of the models')
Train_Models()

print('dumping the files into a pickle file...')
with open("D:/MYWORLD/my academics related/8th sem related/2_PHN_400B_BTP_rel/Models_package.pkl", 'wb') as f:
    pickle.dump(models, f)
    
    
print('Successfully completed the job, terminating the workspace... dot..!^!')