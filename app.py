import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import h5py
from keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import joblib
from sklearn.preprocessing import LabelEncoder
import cv2




app=Flask(__name__)
model1 = pickle.load(open('model.pkl', 'rb'))
model2 = load_model('./audio_notebook/heartbeat_model.h5')
with open('Random_forest_model/random_forest_model.pkl', 'rb') as f:
    model3 = pickle.load(f)




def extract_features_ecg(image_path):
    # Load ECG image
    img = cv2.imread(image_path)

    #Resize image
    target_size=(256,256)
    img = cv2.resize(img,target_size)
    
    # Pre-processing: Perform image enhancement techniques
    img = cv2.medianBlur(img, 5)
    img = cv2.normalize(img, None, 0, 256, cv2.NORM_MINMAX)

    # Segmentation: Locate the ECG waveform in the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    _, thresh = cv2.threshold(img, 127, 256, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ROI = max(contours, key=cv2.contourArea)

    # Resampling: Resample the ECG image to a 1D signal
    y = cv2.polylines(img, [ROI], False, 255, 1)
    y = np.mean(y, axis=0)
    y = y.astype(np.float32)
    x = np.arange(0, len(y))

    # Denoising: Apply denoising techniques
    y = np.convolve(y, np.ones(5)/5, mode='same')
    z=plt.plot(x, y)
    z=plt.plot(x, y)[0]
    xdata = z.get_xdata()
    ydata = z.get_ydata()
    z_array = np.array([xdata, ydata]).T
    
    return z_array

def extract_features(audio_path, offset):
    # y, sr = librosa.load(audio_path, duration=3)
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    # y = librosa.util.normalize(y)

    S = librosa.feature.melspectrogram(
        y, sr=sr, n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    #plt.figure(figsize=(20,10))
    # y1,sr1 = librosa.load(audio_path, duration=3)
    # librosa.display.waveshow(y1, sr=sr1)
    #plt.show()
    #lt.savefig('static/my_wave.jpg')
    #wave.save('static/my_wave.jpg')

    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs

# def wave_signal(audio_path,offset):
#     y, sr = librosa.load(audio_path,offset=offset, duration=3)
#     wave=librosa.display.waveshow(y, sr=sr)
#     wave.save('static/my_wave.jpg')
#     return 'Image generated and saved successfully'

@app.route('/') #path set
def home():
    return render_template('index.html')

@app.route('/index') #path set
def index():
    return render_template('index.html')


@app.route('/report',  methods=['GET', 'POST']) #path set
def about():
    if request.method == 'POST':
        # formData = request.form
        # p1 = str(formData.get('param1'))
        # p2 = str(formData.get('param2'))
        # p3 = str(formData.get('param3'))
        # p4 = str(formData.get('param4'))
        # p5 = str(formData.get('param5'))
        # p6 = str(formData.get('param6'))
        # print("Parameter 1: ",p1)
        # print("Parameter 6: ",p6)
        # return render_template('about.html',p1=p1,p2=2,p3=p3,p4=p4,p5=p5,p6=p6)
        int_param = [float(x) for x in request.form.values()]
        final_param = (np.asarray(int_param)).reshape(1,-1)
        print(int_param)
        print(final_param)
        prediction = model1.predict(final_param)
        P1="Good"
        P2="Not Good" 
        if prediction[0]==0:
            print(P1)
            return render_template('report.html',prediction=P1)
        else:
            print(P2)
            return render_template('report.html',prediction=P2)
        # print("your prediction is",prediction)
        
    return render_template('report.html', prediction=None)

@app.route('/audio', methods=['POST', 'GET'])  # path set
def service():
    if request.method == 'POST':
        audio_file = request.files['file']
        print("the audio file is : ----------", audio_file)
        model = load_model("heartbeat_model.h5")
        
        x_test = []
        x_test.append(extract_features(audio_file, 0.5))
        x_test = np.asarray(x_test)
        x_test = x_test.reshape(
            x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        pred = model.predict(x_test, verbose=1)
        print("pred -->", pred)
    
        heartbeat_status = ""
        confidence = 0
        pred_class = (model.predict(x_test) > 0.5).astype("int32")
        if pred_class[0][1]:
            heartbeat_status = "Normal heartbeat"
            confidence = round(pred[0][1]*100,2,)
        else:
            heartbeat_status = "Abnormal heartbeat"
            confidence = round(pred[0][0]*100,2)
        return render_template(
            'audio.html',
            heartbeat_status=heartbeat_status,
            confidence=confidence,
            
            
           
        )

    return render_template('audio.html', heartbeat_status=None, confidence=0)

@app.route('/ecg', methods=['POST', 'GET'])  # path set
def ecg():
    if request.method == 'POST':
        ecg_file = request.files['ecg_file']
        
        file_path = "R:\MajorProject\Vital-Organ-HealthPredictor-v2-main\Vital-Organ-HealthPredictor-v2-main\static" + ecg_file.filename
        ecg_file.save(file_path)
        print("the ecg file is : ----------", ecg_file)
        print("file path", file_path)
        #ecgmodel = pickle.load("Random_forest_model/random_forest_model_ecg.pkl")
        x_new=[]
        x_new.append(extract_features_ecg(file_path))
        # x_new = np.reshape(x_new,1,512)
        #x_new_flat = np.reshape(x_new, (x_new.shape[0], -1))
        # # Verify the shape of x_train and y_train
        x_new = np.array(x_new)
        print("x_new shape:", x_new.shape)
        x_new_flat = np.reshape(x_new, (x_new.shape[0], -1))
        x_new_flat.shape
        y_pred = model3.predict(x_new_flat)
        print("pred-->",y_pred)
        if y_pred == 1:
            heartecg_status = "Normal"
            print("Normal")
        else:
            heartecg_status = "Abnormal"
            print("Abnormal")
        return render_template(
            'ecg.html',
            heartecg_status=heartecg_status,
            file_path=file_path,
            ecg_file=ecg_file
           
        )

    return render_template('ecg.html', heartecg_status=None, file_path=None, ecg_file=None)




if __name__ == "__main__":
    app.run(debug=True)