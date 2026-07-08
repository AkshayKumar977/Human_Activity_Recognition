import os
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import torch
from model import HAR_LSTM
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)
MODEL_PATH = os.path.join(BASE_DIR,"models","lstm_har_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HAR_LSTM()
model.load_state_dict(torch.load(MODEL_PATH,map_location = device))
model.to(device)
model.eval()
activities = {
    0:'Walking',
    1:'Walking Upstairs',
    2:'Walking Downstairs',
    3:'Sitting',
    4:'Standing',
    5:'Lying'
}
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        files = request.files['file']
        data = pd.read_csv(files,header = None)
        if data.shape != (128,9):
            return render_template('index.html',prediction = "Invalid input. Please upload a CSV file with 128 rows and 9 columns.")
        data = data.to_numpy().reshape(1,128,9)
        data = torch.tensor(data,dtype = torch.float32).to(device)
        with torch.no_grad():
            output = model(data)
            _,predicted = torch.max(output,1)
            prediction = activities[predicted.item()]
            return render_template('index.html',prediction = f"The predicted activity is: {prediction}")
if __name__ == '__main__':
    app.run(debug = True)
