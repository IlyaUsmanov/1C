import argparse
from PIL import Image, ImageDraw
import numpy as np
from math import inf
from itertools import product

def parse_input():
  parser = argparse.ArgumentParser(description='This project allows you to visualize tic-tac-toe victory')
  parser.add_argument('--image_name', help='tic-tac-toe image name')
  parser.add_argument('--width', help='width of lines')
  args = parser.parse_args()
  return args

def convert_picture(image_name):
  with Image.open(image_name) as image:
    matrix = np.array(image)
    return (matrix == [0, 0, 0, 255]).all(axis=2)

def first_nonzero(matrix, axis):
    mask = (matrix != 0)
    return np.where(mask.any(axis = axis), mask.argmax(axis = axis), inf)

def last_nonzero(matrix, axis):
    mask = (matrix != 0)
    value = matrix.shape[axis] - np.flip(mask, axis = axis).argmax(axis = axis) - 1
    return np.where(mask.any(axis = axis), value, -1)

def get_borders(lim1, lim2, lim3, width):
  x1 = np.min(lim2)
  x2 = np.argmin(lim1)
  x3 = x2 + width
  x5 = np.size(lim1) - np.argmin(np.flip(lim1))
  x4 = x5 - width
  x6 = np.max(lim3)
  return np.array([[x1, x2], [x3, x4], [x5, x6]]).astype(int)

def split_on_cells(binary_matrix, width):
  left = first_nonzero(binary_matrix, 1)
  right = last_nonzero(binary_matrix, 1)
  up = first_nonzero(binary_matrix, 0)
  down = last_nonzero(binary_matrix, 0)
  x = get_borders(left, up, down, width)
  y = get_borders(up, left, right, width)
  return (x, y)

def get_types(binary_matrix, cell_coordinates):
  types = np.empty((3, 3))
  for i in range(3):
    for j in range(3):
      cell = (cell_coordinates[0][i], cell_coordinates[1][j])
      x_mid = (cell[0][0] + cell[0][1]) // 2
      y_mid = (cell[1][0] + cell[1][1]) // 2
      black_pixels = np.sum(binary_matrix[cell[0][0]:cell[0][1], cell[1][0]:cell[1][1]])
      if (binary_matrix[x_mid][y_mid] == 1):
        types[i][j] = 1
      elif black_pixels != 0:
        types[i][j] = 2
      else:
        types[i][j] = 0 
  return types     

def find_line(cell_types, coords):
  for i in range(3):
    cell_types_horizontal = cell_types[i,:]
    if (np.all(cell_types_horizontal == cell_types[i][0]) and cell_types[i][0] != 0):
      cell_mid = (coords[0][i][0] + coords[0][i][1]) // 2
      return (coords[1][0][0], cell_mid, coords[1][2][1], cell_mid)
  for i in range(3):
    cell_types_vertical = cell_types[:,i]
    if (np.all(cell_types_vertical == cell_types[0][i]) and cell_types[0][i] != 0):
      cell_mid = (coords[1][i][0] + coords[1][i][1]) // 2
      return (cell_mid, coords[0][0][0], cell_mid, coords[0][2][1])
  diagonal = np.diagonal(cell_types)
  if (np.all(diagonal == cell_types[0][0]) and cell_types[0][0] != 0):
    return (coords[1][0][0], coords[0][0][0], coords[1][2][1], coords[0][2][1])
  diagonal = np.fliplr(cell_types).diagonal()
  if (np.all(diagonal == cell_types[2][0]) and cell_types[2][0] != 0):
    return (coords[1][0][0], coords[0][2][1], coords[1][2][1], coords[0][0][0])
  return None

def draw_line(line_coordinates, image_name, width):
  with Image.open(image_name) as image:
    draw = ImageDraw.Draw(image)
    draw.line(line_coordinates, fill=(0, 0, 0), width = width)
    image.save("result.png", "PNG")

def create_output(args):
  args.width = int(args.width)
  binary_matrix = convert_picture(args.image_name)
  cell_coordinates = split_on_cells(binary_matrix, args.width)
  cell_types = get_types(binary_matrix, cell_coordinates)
  line_coordinates = find_line(cell_types, cell_coordinates)
  if line_coordinates == None:
    print('No winning_line')
  else:
    draw_line(line_coordinates, args.image_name, args.width)

def main():
  args = parse_input()
  create_output(args)

if __name__ == '__main__':
  main()
