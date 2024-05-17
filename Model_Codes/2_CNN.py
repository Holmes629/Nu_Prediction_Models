import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Layer
from keras.models import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os

print('Initializing the job..... Setting up the workspace...')
print('Loading the datasets...')

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "extracted_data_csv_3.csv")
data = pd.read_csv(file_path)

print('Processing the data....')
Pr = [np.log(i) for i in data["Pr "]]
Ra = [np.log(i) for i in data['Ra']]
Nu = [float(i) for i in data['Nu']]
Aspect_ratio = [float(i) for i in data['Aspect_ratio']]
Shape = [float(i) for i in data['Shape']]
Boundary_cond = [float(i) for i in data['Boundary_conditions']]

All_combined = pd.DataFrame({'Pr ': Pr, 'Ra': Ra, 'Aspect_ratio': Aspect_ratio, 'Shape': Shape, 'Boundary_conditions': Boundary_cond})
Nu = pd.DataFrame({'Nu': Nu})
# X_train, X_test, y_train, y_test = train_test_split(All_combined, Nu, test_size=0.2, random_state=0)
X_train= All_combined
y_train= Nu
print('Data Processing completed....Loading the models....')

# Convolutional Neural Network (CNN)
cnn_model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='relu')   # Output layer with linear activation for regression
])


models = {'CNN': cnn_model}

Predictions = dict()
print('Models loaded..... Initializing training.......')

# Model Training
for name, model in models.items():
    model.compile(optimizer='adam', loss='mean_squared_error')
for name, model in models.items():
    print(f'started training of {name} model: ')
    hist = model.fit(X_train, y_train, epochs=5000)
    plt.plot(hist.history['loss'], label='Train Loss')
    # plt.plot(hist.history['val_loss'], label="Validation loss")
    plt.title(f'{name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    # Predictions[name] = model.predict(X_test).flatten()
    print(f'Training and Saving of {name} completed....')

# Testing and Plotting Predictions
# for name, model in models.items():
#     pred = pd.DataFrame({'Nu_test': y_test['Nu'], 'Nu_pred_fnn': Predictions[name]})
#     sns.lineplot(data= pred)
#     plt.xlabel('No. of data points')
#     plt.ylabel('Nu_Actual and Predicted')
#     plt.title(f'{name} predictions plot')
#     plt.show()
#     # The mean squared error and R^2
#     print('Mean squared error: %.2f'
#         % mean_squared_error(y_test, Predictions[name]))
#     print('R^2 Score: %.2f'
#         % r2_score(y_test, Predictions[name]))
    
# for saving the packages
print('Saving the models into .h5 packages...')
for name, mode in models.items():
    package_path= os.path.join(current_directory, f"{name}_Model_Package.h5")
    model.save(package_path)
    print(f"{name} model saved successfully....")  
    
print('Job completed.... Terminating the program....')