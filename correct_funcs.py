#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np

#correct function
def correct(correction, puzzle):
    '''
    Takes input (i,j,k) corresponding to manual correction of entry in ith row, jth column to digit k
    for the other input puzzle (a 9 by 9 numpy array corresponding to a sudoku puzzle)
    and returns the modified puzzle
    '''
    i, j, k = correction
    puzzle[i-1][j-1] = k

    return puzzle