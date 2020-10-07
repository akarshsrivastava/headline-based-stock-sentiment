# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('classifier.pkl', 'rb')) # loading the trained model
vec = pickle.load(open('vec.pkl','rb')) # loading the tfidf vectorizer

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    label=""
    # retrieving values from form
    if request.method == "POST":
        name = request.form['name']
        headline = request.form['headline']
        #head = re.sub(r'[^a-zA-Z]', ' ', headline)
        #head = head.lower()
        #head = re.sub(' +', ' ', head)
        data = [headline]
        vect = vec.transform(data).toarray()
        pred = model.predict(vect)
        
        if int(pred)== 1: 
            prediction ='increasing stock movement predicted for ' + name
        if int(pred)==0: 
            prediction ='no strong movement predicted for ' + name
        if int(pred)==-1:
            prediction ='decreasing or no stock movement predicted for ' + name
        
        return render_template("index.html", prediction = prediction)
        #label = str(pred)

#         output=""    

#         if(label == '1'):
#             output = 'increasing stock movement predicted for ' + name 

#         if(label == '0'):
#             output = 'decreasing or no stock movement predicted for ' + name

#         if(label == '-1'):
#             output = 'decreasing or no stock movement predicted for ' + name
    
    #return render_template('index.html', prediction='Prediction: {}'.format(label)) # rendering the predicted result
    
if __name__ == "__main__":
    app.run(debug=True)