from flask import Flask, g, render_template, request

import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from PIL import Image

#import or own sudoku modules
import print_funcs as pf
import prep_and_predict as pnp
import solve_funcs as sf


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import imread

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
            img = request.files['image'] #get image as jpg object

            error1 =False
            error2 =False
            error3 =False


            
            try:
                img = Image.open(img)
                gray_img = img.convert("L")
                img = np.array(gray_img)
            except:
                error1 = True
                return render_template('submit.html', error1=True)
            #np.loadtxt(img) # numpy array with the pixel values
            img_shape = img.shape
            
            # 2 load pickle model, call CompleteSudokuPredictFromRaw to get prediction of puzzle from the model
            

            try:
                model = pickle.load(open('sudoku-model/model.pkl', 'rb'))
            
                
            except:
                return render_template('submit.html', error2=True)
            # 5. this code is for displaying an image, we want to print a numpy array 
            #fig = Figure(figsize=(3, 3))
            #ax = fig.add_subplot(1, 1, 1,)
            #ax.imshow(img, cmap='binary')
            #ax.axis("off")

            # weird part of 5. 
            #pngImage = io.BytesIO()
            #FigureCanvas(fig).print_png(pngImage) # convert the pyplot figure object to a PNG image

            # encode the PNG image to base64 string
            #pngImageB64String = "data:image/png;base64,"
            #pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

            #instead, create string for predicted puzzle, and solution
            puzzle, uncertain_values = pnp.CompleteSudokuPredictFromRaw(img, model)
            
            puzzle_str = pf.print_puzzle(puzzle)

            if len(uncertain_values) > 0:
                uncertain_str = "We are uncertain if our prediction(s) in"
                for i in range(len(uncertain_values)):
                    #add comma if not first value
                    if i > 0:
                        uncertain_str += ","
                    
                    cell = uncertain_values[i]
                    uncertain_str += " row " + str(cell[0]) + ", column " + str(cell[1])
                    
                    
                uncertain_str += " is correct.\n Please double check this with the image."

                return render_template('submit.html', uncertainty=True, prediction=puzzle_str, uncertain=uncertain_str)
            
            else:
                

                #compute solution
                puzzle_sol = puzzle.copy()
                try:
                    #try to solve the puzzle
                    sf.sudoku_solve(puzzle_sol)
                    puzzle_sol_str = pf.print_puzzle(puzzle_sol)
                    return render_template('submit.html', prediction=puzzle_str, solution=puzzle_sol_str)
                except:
                    #if cannot solve, only return prediction
                    return render_template('submit.html', prediction=puzzle_str, solve_error = True)
            
        except:
            return render_template('submit.html', error=True)

'''
@app.route('/correcting', methods=['POST'])
def correcting():
    
    try:
    
        corr = request.form['correction'] #get correction      
        return render_template('submit.html', adjusting=True, adjustment = corr)
    except:
        return render_template('submit.html', bigerror = True)
        '''