from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

print('Initializing the job..... Setting up the workspace...')
print('Loading the datasets...')
data = pd.read_csv("extracted_data_csv_3.csv")

print('Processing the data....')
Pr = data["Pr "]  # Assuming the column name is 'Pr'
Ra = data['Ra']
Nu = data['Nu']
Aspect_ratio = data['Aspect_ratio']
Shape = data['Shape']
Boundary_cond = data['Boundary_conditions']

Pr= [(np.log(i)) for i in Pr]
Ra= [(np.log(i)) for i in Ra]
Nu= [int(i) for i in Nu]
Aspect_ratio= [int(i) for i in Aspect_ratio]
Shape= [int(i) for i in Shape]
Boundary_cond= [int(i) for i in Boundary_cond]
All_combined = pd.DataFrame({'Pr ': Pr, 'Ra': Ra, 'Aspect_ratio': Aspect_ratio, 'Shape': Shape, 'Boundary_conditions': Boundary_cond})
Nu= pd.DataFrame({'Nu': Nu})

X_train_reshaped = All_combined.values.reshape(All_combined.shape[0], All_combined.shape[1], 1)
print('Data Processing completed....Loading the models....')

# Feedforward Neural Network (FNN)
fnn_model= keras.Sequential([
    keras.layers.Flatten(input_shape=(All_combined.shape[1], 1)),
    keras.layers.Dense(4, activation='linear'),
    keras.layers.Dense(4, activation='linear'),
    keras.layers.Dense(1, activation='linear')
])

# Convolutional Neural Network (CNN)
cnn_model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='linear', input_shape=(All_combined.shape[1], 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='linear'),
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])

# Recurrent Neural Network (RNN (LSTM))
rnn_model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(All_combined.shape[1], 1)),
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])

models= {'FNN': fnn_model, 
         'CNN': cnn_model, 
         'RNN': rnn_model}
print('Models loaded..... Initializing training.......')

for name, model in models.items():
    model.compile(optimizer='adam', loss='mean_squared_error', metrics= ['accuracy'])

# Train and evaluate models (replace with your actual training and evaluation code)
for name, model in models.items():
    print(f'started training of {name} model: ')
    model.fit(All_combined, Nu, epochs= 20)
    print('dumping the files into a package file...')
    model.save(f"D:/MYWORLD/my academics related/8th sem related/2_PHN_400B_BTP_rel/MLAPI/Model_Packages/{name}_Model_Package.h5")
    print(f'        Training and Saving of {name} completed....')
    
# print('Successfully completed the job, terminating the workspace... dot..!^!')
# fnn_model_load= keras.models.load_model("D:/MYWORLD/my academics related/8th sem related/2_PHN_400B_BTP_rel/MLAPI/Model_Packages/FNN_Model_Package.h5")
# cnn_model_load= keras.models.load_model("D:/MYWORLD/my academics related/8th sem related/2_PHN_400B_BTP_rel/MLAPI/Model_Packages/CNN_Model_Package.h5")
# rnn_model_load= keras.models.load_model("D:/MYWORLD/my academics related/8th sem related/2_PHN_400B_BTP_rel/MLAPI/Model_Packages/RNN_Model_Package.h5")

# print(fnn_model_load.predict(X_train_reshaped))
# print(cnn_model_load.predict(X_train_reshaped))
# print(rnn_model_load.predict(X_train_reshaped))