from flask import Flask, g, render_template, request

import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import pickle

#import or own sudoku modules
import print_funcs
import cnn_predict_funcs as cnn
import preprocess_funcs as prep
import solve_funcs


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io
import base64

### stuff from last class
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main_better.html')


# matplotlib: https://matplotlib.org/3.5.0/gallery/user_interfaces/web_application_server_sgskip.html
# plotly: https://towardsdatascience.com/web-visualization-with-plotly-and-flask-3660abf9c946

@app.route('/submit-sudoku/', methods=['POST', 'GET'])
def submit():
    if request.method == 'GET':
        return render_template('submit.html')
    else:
        try:            
            '''
            1. Access the image
            2. Load the pickled ML model
            3. Run the ML model on the image
            4. Store the ML model's prediction in some Python variable
            5. Show the image on the template
            6. Print the prediction and some message on the template
            '''
            # 1
            img = request.files['image'] # jpg object
            img = np.loadtxt(img) # numpy array with the pixel values

            
            # 2 load pickle model, call CompleteSudokuPredictFromRaw to get prediction of puzzle from the model
            model = pickle.load(open('sudoku-model/model.pkl', 'rb'))
            
            # 3, 4
            puzzle = cnn.CompleteSudokuPredictFromRaw(img, model)

            # 5. NOTE this code is for displaying an image, we want to print a numpy array 
            fig = Figure(figsize=(3, 3))
            ax = fig.add_subplot(1, 1, 1,)
            ax.imshow(img, cmap='binary')
            ax.axis("off")

            # weird part of 5. 
            pngImage = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage) # convert the pyplot figure object to a PNG image

            # encode the PNG image to base64 string
            pngImageB64String = "data:image/png;base64,"
            pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

            return render_template('submit.html',
             image=pngImageB64String, digit=digit)
        except:
            return render_template('submit.html', error=True)
