#!/usr/bin/env python
# coding: utf-8

# In[1]:


#printpuzzle() function
def print_puzzle(p):
    '''
    Takes input p a sudoku puzzle as a 9 by 9 numpy array,
    and returns a string giving a realistic depiction of a sudoku puzzle
    '''
    puzzleprint = ""
    for i in range(len(p)):
        if i % 3 == 0 and i != 0:
            puzzleprint += "------------------------ \n"

        for j in range(len(p[0])):
            if j % 3 == 0 and j != 0:
                puzzleprint += " | "

            if j == 8:
                puzzleprint += str(p[i][j]) + "\n"
            else:
                puzzleprint += str(p[i][j]) + " "
    return puzzleprint

