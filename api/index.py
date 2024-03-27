from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the paths for model files using the current directory
# models_path = os.path.join(current_directory, "Model_Packages", "ML_Models_Package.pkl")
# fnn_model_path = os.path.join(current_directory, "Model_Packages", "FNN_Model_Package.h5")
# cnn_model_path = os.path.join(current_directory, "Model_Packages", "CNN_Model_Package.h5")
# rnn_model_path = os.path.join(current_directory, "Model_Packages", "RNN_Model_Package.h5")

# with open(models_path, 'rb') as f:
#     models= pickle.load(f)
# fnn_model= keras.models.load_model(fnn_model_path)
# cnn_model= keras.models.load_model(cnn_model_path)
# rnn_model= keras.models.load_model(rnn_model_path)
app= Flask(__name__)

@app.route('/')
def home():
    print("working")
    return render_template('index.html')

# @app.route('/predict', methods= ['POST'])
# def predict():
#     int_features= [float(x) for x in request.form.values()]
#     final_features1= pd.DataFrame({'Pr ': [int_features[0]], 'Ra': [int_features[1]], 'Aspect_ratio': [int_features[2]], 'Shape': [int_features[3]], 'Boundary_conditions': [int_features[4]]})
#     final_features2= pd.DataFrame({'Pr ': [np.log(int_features[0])], 'Ra': [np.log(int_features[1])], 'Aspect_ratio': [int_features[2]], 'Shape': [int_features[3]], 'Boundary_conditions': [int_features[4]]})
#     final_features3 = final_features2.values.reshape(final_features2.shape[0], final_features2.shape[1], 1)
    
#     prediction1= models['1 Linear Regression'].predict(final_features1)
#     prediction2= models['1 SVM'].predict(final_features1)
#     prediction3= models['1 Random Forest'].predict(final_features1)
#     prediction4= models['2 Log Linear Regression'].predict(final_features2)
#     prediction5= models['2 Log SVM'].predict(final_features2)
#     prediction6= models['2 Log Random Forest'].predict(final_features2)
#     prediction7= fnn_model.predict(final_features3)
#     prediction8= cnn_model.predict(final_features3)
#     prediction9= rnn_model.predict(final_features3)
    
#     output1= round(prediction1[0], 2)
#     output2= round(prediction2[0], 2)
#     output3= round(prediction3[0], 2)
#     output4= round(prediction4[0], 2)
#     output5= round(prediction5[0], 2)
#     output6= round(prediction6[0], 2)
#     output7= round(prediction7[0][0], 2)
#     output8= round(prediction8[0][0], 2)
#     output9= round(prediction9[0][0], 2)
    
#     return render_template('index.html', prediction_text1= output1,
#                            prediction_text2= output2,
#                            prediction_text3= output3,
#                            prediction_text4= output4,
#                            prediction_text5= output5,
#                            prediction_text6= output6,
#                            prediction_text7= output7,
#                            prediction_text8= output8,
#                            prediction_text9= output9)

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data= request.get_json(force=True)
#     final_features= [np.array(list(data.values()))]
#     # final_features2= [np.array([np.log(int_features[0]), np.log(int_features[1]), int_features[2], int_features[3], int_features[4]])]

#     prediction1= models['1 Linear Regression'].predict(final_features)
#     prediction2= models['1 SVM'].predict(final_features)
#     prediction3= models['1 Random Forest'].predict(final_features)
#     prediction4= models['2 Log Linear Regression'].predict(final_features)
#     prediction5= models['2 Log SVM'].predict(final_features)
#     prediction6= models['2 Log Random Forest'].predict(final_features)
#     prediction7= fnn_model.predict(final_features)
#     prediction8= cnn_model.predict(final_features)
#     prediction9= rnn_model.predict(final_features)
    
#     output1= round(prediction1[0], 2)
#     output2= round(prediction2[0], 2)
#     output3= round(prediction3[0], 2)
#     output4= round(prediction4[0], 2)
#     output5= round(prediction5[0], 2)
#     output6= round(prediction6[0], 2)
#     output7= round(prediction7[0], 2)
#     output8= round(prediction8[0], 2)
#     output9= round(prediction9[0], 2)
    
#     return jsonify(output1, output2, output3, output4, output5, output6, output7, output8, output9)


if __name__=='__main__':
    app.run(debug=False)
