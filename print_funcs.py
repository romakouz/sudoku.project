#!/usr/bin/env python
# coding: utf-8

# In[1]:


#printpuzzle() function
def print_puzzle(p):
  '''
  Takes input p a sudoku puzzle as a 9 by 9 numpy array,
  and prints a more realistic depiction of a sudoku puzzle
  '''
  for i in range(len(p)):
      if i % 3 == 0 and i != 0:
          print("------------------------ ")

      for j in range(len(p[0])):
          if j % 3 == 0 and j != 0:
              print(" | ", end="")

          if j == 8:
              print(p[i][j])
          else:
              print(str(p[i][j]) + " ", end="")

