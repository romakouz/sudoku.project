#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# sudoku predict functions (using our CNN model) (and dependencies)

import tensorflow as tf
import numpy as np

#softmax function
softmax = tf.keras.layers.Softmax()

#CNN predict function
def CNN1predict(image, model_pickled):
  '''
  returns a tuple containing (digit prediction, probability of certainty of prediction)
  for input: greyscale image of shape (50,50,1) or (50,50),
  using our CNN model1
  '''
  #read in the pickled model as model1
  model1 = model_pickled

  imager = image.reshape(1, 50, 50, 1)
  #call softmax on prediction to get tf array containing prob distribution across 10 digits
  pred_vect = softmax(model1.predict(imager))
  #change to numpy 10 element vector
  pred_vect = pred_vect.numpy()
  pred_vect = pred_vect.reshape(pred_vect.shape[1])
  #get prediction and probability of prediction
  pred_index = np.argmax(pred_vect)
  pred_prob = pred_vect[pred_index]
  return pred_index, pred_prob

#sudoku puzzle predict function (from 450*450 image)
def CompleteSudokuPredict(image):
  '''
  Inputs an image of a full sudoku puzzle (after preprocessing), in 450*450 np array, and proceeds with
  calling CNN on each cell to return a numpy array of the predicted puzzle 
  '''

  puzzle_arr =np.zeros((9,9),dtype=np.int64)

  cells = splitcells(image) #get list of all cells in image
  for i in range(9):
    for j in range(9):
      prediction, certainty = CNN1predict(cells[i*9 +j])
      puzzle_arr[i][j]=prediction
      if certainty < 0.5: 
        print("We are uncertain if the prediction in row " + i.str() + ", column "+ j.str() +" is correct.")
  
  return puzzle_arr

#sudoku puzzle predict function
def CompleteSudokuPredictFromRaw(image, model_pickled):
  '''
  Inputs an image of a full sudoku puzzle (before preprocessing), applies preprocessing,
  converts into 450*450 np array, and proceeds with calling CNN on each cell,
  returns a numpy array of the predicted puzzle 
  '''
  image = fullpreprocess(image)
  puzzle_arr =np.zeros((9,9),dtype=np.int64)

  cells = splitcells(image) #get list of all cells in image
  for i in range(9):
    for j in range(9):
      prediction, certainty = CNN1predict(cells[i*9 +j], model_pickled)
      puzzle_arr[i][j]=prediction
      if certainty < 0.5: 
        print("We are uncertain if the prediction in row " + i.str() + ", column "+ j.str() +" is correct.")
  
  return puzzle_arr

