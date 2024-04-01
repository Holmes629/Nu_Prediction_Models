from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
# from tensorflow import keras
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the paths for model files using the current directory
lin_reg_path = os.path.join(current_directory, "Model_Packages", "1_Linear_Regression_Model_Package.pkl")
svm_path = os.path.join(current_directory, "Model_Packages", "2_SVM_Model_Package.pkl")
# models_path = os.path.join(current_directory, "Model_Packages", "3_Random_Forest_Model_Package.pkl")
log_lin_reg_path = os.path.join(current_directory, "Model_Packages", "2_Log_Linear_Regression_Model_Package.pkl")
log_svm_path= os.path.join(current_directory, "Model_Packages", "2_Log_SVM_Model_Package.pkl")
# models_path = os.path.join(current_directory, "Model_Packages", "2_Random_Forest_Model_Package.pkl")
# fnn_model_path = os.path.join(current_directory, "Model_Packages", "FNN_Model_Package.h5")
# cnn_model_path = os.path.join(current_directory, "Model_Packages", "CNN_Model_Package.h5")
# rnn_model_path = os.path.join(current_directory, "Model_Packages", "RNN_Model_Package.h5")

with open(lin_reg_path, 'rb') as f:
    lin_reg_model= pickle.load(f)
with open(svm_path, 'rb') as f:
    svm_model= pickle.load(f)
with open(log_lin_reg_path, 'rb') as f:
    log_lin_reg_model= pickle.load(f)
with open(log_svm_path, 'rb') as f:
    log_svm_model= pickle.load(f)
# fnn_model= keras.models.load_model(fnn_model_path)
# cnn_model= keras.models.load_model(cnn_model_path)
# rnn_model= keras.models.load_model(rnn_model_path)
app= Flask(__name__)

@app.route('/')
def home():
    print("working")
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    final_features1= pd.DataFrame({'Pr ': [int_features[0]], 'Ra': [int_features[1]], 'Aspect_ratio': [int_features[2]], 'Shape': [int_features[3]], 'Boundary_conditions': [int_features[4]]})
    final_features2= pd.DataFrame({'Pr ': [np.log(int_features[0])], 'Ra': [np.log(int_features[1])], 'Aspect_ratio': [int_features[2]], 'Shape': [int_features[3]], 'Boundary_conditions': [int_features[4]]})
    final_features3 = final_features2.values.reshape(final_features2.shape[0], final_features2.shape[1], 1)
    
    prediction1= lin_reg_model.predict(final_features1)
    prediction2= svm_model.predict(final_features1)
    # prediction3= models['1 Random Forest'].predict(final_features1)
    prediction4= log_lin_reg_model.predict(final_features2)
    prediction5= log_svm_model.predict(final_features2)
    # prediction6= models['2 Log Random Forest'].predict(final_features2)
    # prediction7= fnn_model.predict(final_features3)
    # prediction8= cnn_model.predict(final_features3)
    # prediction9= rnn_model.predict(final_features3)
    
    output1= round(prediction1[0], 2)
    output2= round(prediction2[0], 2)
    # output3= round(prediction3[0], 2)
    output4= round(prediction4[0], 2)
    output5= round(prediction5[0], 2)
    # output6= round(prediction6[0], 2)
    # output7= round(prediction7[0][0], 2)
    # output8= round(prediction8[0][0], 2)
    # output9= round(prediction9[0][0], 2)
    
    return render_template('index.html', prediction_text1= output1,
                           prediction_text2= output2,
                           prediction_text3= 0,
                           prediction_text4= output4,
                           prediction_text5= output5,
                           prediction_text6= 0,
                           prediction_text7= 0,
                           prediction_text8= 0,
                           prediction_text9= 0)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.get_json(force=True)
    final_features= [np.array(list(data.values()))]
    final_features2= [np.array([np.log(final_features[0]), np.log(final_features[1]), final_features[2], final_features[3], final_features[4]])]

    # prediction1= models['1 Linear Regression'].predict(final_features)
    # prediction2= models['1 SVM'].predict(final_features)
    # prediction3= models['1 Random Forest'].predict(final_features)
    # prediction4= models['2 Log Linear Regression'].predict(final_features2)
    # prediction5= models['2 Log SVM'].predict(final_features2)
    # prediction6= models['2 Log Random Forest'].predict(final_features2)
    # prediction7= fnn_model.predict(final_features2)
    # prediction8= cnn_model.predict(final_features2)
    # prediction9= rnn_model.predict(final_features2)
    
    # output1= round(prediction1[0], 2)
    # output2= round(prediction2[0], 2)
    # output3= round(prediction3[0], 2)
    # output4= round(prediction4[0], 2)
    # output5= round(prediction5[0], 2)
    # output6= round(prediction6[0], 2)
    # output7= round(prediction7[0], 2)
    # output8= round(prediction8[0], 2)
    # output9= round(prediction9[0], 2)
    return jsonify()
    # return jsonify(output1, output2, output3, output4, output5, output6, output7, output8, output9)


if __name__=='__main__':
    app.run(debug=False)
