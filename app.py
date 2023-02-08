import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import h5py
from keras.models import load_model
import librosa


app=Flask(__name__)
model1 = pickle.load(open('model.pkl', 'rb'))
model2 = load_model('./audio_notebook/heartbeat_model.h5')

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

    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs


@app.route('/') #path set
def home():
    return render_template('index.html')

@app.route('/index') #path set
def index():
    return render_template('index.html')


@app.route('/about',  methods=['GET', 'POST']) #path set
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
            return render_template('about.html',prediction=P1)
        else:
            print(P2)
            return render_template('about.html',prediction=P2)
        # print("your prediction is",prediction)
        
    return render_template('about.html', prediction=None)

@app.route('/service', methods=['POST', 'GET'])  # path set
def service():
    if request.method == 'POST':
        audio_file = request.files['file']
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
            confidence = pred[0][1]
        else:
            heartbeat_status = "Abnormal heartbeat"
            confidence = pred[0][0]
        return render_template(
            'service.html',
            heartbeat_status=heartbeat_status,
            confidence=confidence
        )

    return render_template('service.html', heartbeat_status=None, confidence=0)






if __name__ == "__main__":
    app.run(debug=True)