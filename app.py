import string
from flask import Flask, render_template, request
import pickle
import numpy as np


app=Flask(__name__)
model1 = pickle.load(open('model.pkl', 'rb'))

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

@app.route('/service',methods=['POST','GET']) #path set
def service():
    print(type(request.form.values))
    return render_template('service.html')





if __name__ == "__main__":
    app.run(debug=True)