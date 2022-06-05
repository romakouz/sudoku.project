#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#solve function (and dependencies)

#valid function
def valid(puzzle, number, position):
  '''
  Checks if number is valid for position in puzzle
  (by checking if it appears in the row, column, or subgrid)
  where: - puzzle is given as a 9 by 9 numpy array
        - number is a digit between 1 and 9
  and   - position is a tuple of the form (i, j) representing the position in the ith row and jth column
  '''
  # Check row
  for i in range(len(puzzle[0])):
      if puzzle[position[0]][i] == number and position[1] != i:
          return False

  # Check column
  for i in range(len(puzzle)):
      if puzzle[i][position[1]] == number and position[0] != i:
          return False

  # Check subgrid
  subgrid_x = position[1] // 3
  subgrid_y = position[0] // 3

  for i in range(subgrid_y*3, subgrid_y*3 + 3):
      for j in range(subgrid_x * 3, subgrid_x*3 + 3):
          if puzzle[i][j] == number and (i,j) != position:
              return False

  return True


#find_cell function
def find_cell(puzzle):
  '''
  finds the next empty cell (i.e. location of first'0') in given puzzle
  searching through each row from left to right
  '''
  for i in range(len(puzzle)):
        for j in range(len(puzzle[0])):
            if puzzle[i][j] == 0:
                return (i, j)  # row, col
                #print(i, j)



#solve function
def solve(puzzle):
  '''
  recursively solves puzzle until all cells are filled in (validly)
  using backtracking
  '''
  #find next empty cell, if none are empty, puzzle is solved
  find = find_cell(puzzle)
  if not find:
      return True
  else:
      #row, col store the location of the empty cell
      row, col = find

  for i in range(1,10):
      #check if given entry is valid at location row, col
      if valid(puzzle, i, (row, col)):
          #if valid, input given entry at location
          puzzle[row][col] = i
          #check if we can solve the puzzle with the entry we used
          if solve(puzzle):
              return puzzle, True
          #if we cannot, reset entry to 0, try again
          puzzle[row][col] = 0

  return False

