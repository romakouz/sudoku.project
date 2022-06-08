#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# sudoku predict functions (using our CNN model) (and dependencies)
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

#preprocess function (greyscales/blurs) 
def preprocess(image):
  '''
  This function turns given image to grayscale, blur, 
  and changes the receptive threshold of the image.
  Grayscale currently commented out as this function is only used on grayscale images.
  '''
  
  #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(image, (3,3),6)
  threshold_img = cv2.adaptiveThreshold(blur, 255,1,1,11,2)
  return threshold_img

#main_outline function 
def main_outline(contour):
  '''
  Function to crop and well-aligned suduko by reshaping it,
  using a given detected contour of the images
  '''
  biggest = np.array([])
  max_area = 0
  for i in contour:
    area = cv2.contourArea(i)
    if area > 50:
      peri = cv2.arcLength(i, True)
      approx = cv2.approxPolyDP(i, 0.02 * peri, True)
      if area > max_area and len(approx) == 4:
        biggest = approx
        max_area = area
  return biggest, max_area

#reframe function
def reframe(points):
  '''
  Function to reframe a given contour 
  '''
  points = points.reshape((4,2))
  points_new = np.zeros((4,1,2), dtype = np.int32)
  add = points.sum(1)
  points_new[0] = points[np.argmin(add)]
  points_new[3] = points[np.argmax(add)]
  diff = np.diff(points, axis =1)
  points_new[1] = points[np.argmin(diff)]
  points_new[2] = points[np.argmax(diff)]
  return points_new

#splitcells function
def splitcells(img):
  '''

  '''
  rows = np.vsplit(img, 9)
  boxes = []
  for r in rows:
    cols = np.hsplit(r, 9)
    for box in cols:
      boxes.append(box)
  return boxes

#splitcells_meta function    
def splitcells_meta(data):
  box_vals = []
  for i in range(len(data)):
    for j in range(len(data[0])):
      box_vals.append(data[i][j])
  return box_vals

#complete preprocessing function
def fullpreprocess(img):
    '''
    Given an image of a sudoku puzzle of any size, this function returns a modified 450*450 image
    which has been aligned and fitted to the frame.
    '''
    #changing the image shape to be 450,450. should also make the img variable name pick a random puzzle for later on
    img = cv2.resize(img, (450,450))

    threshold = preprocess(img)

    #detecting contour
    #finding the outline of the sudoku puzzle in the image
    contour_1 = img.copy()
    contour_2 = img.copy()
    contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contour, -1, (0, 255, 0),3)

    #testing
    black_img = np.zeros((450,450,3), np.uint8)
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
      biggest = reframe(biggest)
      cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
      pts1 = np.float32(biggest)
      pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
      matrix = cv2.getPerspectiveTransform(pts1,pts2)

      imagewrap = cv2.warpPerspective(img,matrix,(450,450))
      return imagewrap
    else:
      raise ValueError("Error: Image cannot be preprocessed")



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

  Keep track of any uncertain indices in a list
  '''
  image = fullpreprocess(image)
  puzzle_arr =np.zeros((9,9),dtype=np.int64)
  uncertain = []

  cells = splitcells(image) #get list of all cells in image
  for i in range(9):
    for j in range(9):
      prediction, certainty = CNN1predict(cells[i*9 +j], model_pickled)
      puzzle_arr[i][j]=prediction
      if certainty < 0.4: 
        #add uncertain tuple to list
        unertain.append((i,j))
        print("We are uncertain if the prediction in row " + i.str() + ", column "+ j.str() +" is correct.")
  
  return puzzle_arr, uncertain

