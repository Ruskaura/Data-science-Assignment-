# 1. Import the numpy package under the name np (★☆☆)

import numpy as np

### 2. Print the numpy version and the configuration (★☆☆)

import numpy as np

print("NumPy version:", np.__version__)
print("\nNumPy configuration:")
print(np.show_config())


# 3. Create a null vector of size 10 (★☆☆)

import numpy as np

null_vector = np.zeros(10)
print(null_vector)


# 4. How to find the memory size of any array (★☆☆)
import numpy as np

arr = np.zeros((5, 3))  # Example array
memory_size = arr.nbytes
print("Memory size of the array:", memory_size, "bytes")

# 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

python -c "import numpy as np; help(np.add)"


# 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

import numpy as np

null_vector = np.zeros(10)
null_vector[4] = 1
print(null_vector)

# 7. Create a vector with values ranging from 10 to 49 (★☆☆)

import numpy as np

vector = np.arange(10, 50)
print(vector)

# 8. Reverse a vector (first element becomes last) (★☆☆)

import numpy as np

vector = np.arange(10)  # Example vector
reversed_vector = np.flip(vector)
print(reversed_vector)

# 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

import numpy as np

matrix = np.arange(9).reshape(3, 3)
print(matrix)


# 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)

import numpy as np

arr = np.array([1, 2, 0, 0, 4, 0])
non_zero_indices = np.nonzero(arr)
print(non_zero_indices)


# 11. Create a 3x3 identity matrix (★☆☆)

import numpy as np

identity_matrix = np.eye(3)
print(identity_matrix)

# 12. Create a 3x3x3 array with random values (★☆☆)

import numpy as np

random_array = np.random.rand(3, 3, 3)
print(random_array)

# 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

import numpy as np

# Create a 10x10 array with random values


array = np.random.rand(10, 10)

# Find the minimum and maximum values
min_value = np.min(array)
max_value = np.max(array)

print("Minimum value:", min_value)
print("Maximum value:", max_value)

# 14. Create a random vector of size 30 and find the mean value (★☆☆)

import numpy as np

# Create a random vector of size 30
random_vector = np.random.rand(30)

# Find the mean value
mean_value = np.mean(random_vector)

print("Mean value:", mean_value)


# 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

import numpy as np

# Define the shape of the array
shape = (5, 5)  # Example shape

# Create an array of zeros
array = np.zeros(shape)

# Set the border values to 1
array[0, :] = 1  # Top row
array[-1, :] = 1  # Bottom row
array[:, 0] = 1  # Left column
array[:, -1] = 1  # Right column

print(array)

# 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

import numpy as np

# Example existing array
existing_array = np.ones((3, 3))  # Example array filled with ones

# Create a new array with zeros around the existing array
new_shape = tuple(np.array(existing_array.shape) + 2)  # Increase the shape by 2 in each dimension
new_array = np.zeros(new_shape)
new_array[1:-1, 1:-1] = existing_array  # Insert the existing array into the center of the new array

print(new_array)

# 17. What is the result of the following expression? (★☆☆)
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1

'0 * np.nan results in np.nan'
'np.nan == np.nan' results in 'False'
'np.inf > np.nan' results in 'False'
'np.nan - np.nan' results in 'np.nan'
'np.nan in set([np.nan])' results in 'True'
'0.3 == 3 * 0.1' results in 'False'

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

import numpy as np

# Create a 5x5 matrix filled with zeros
matrix = np.zeros((5, 5))

# Assign values 1, 2, 3, 4 just below the diagonal
np.fill_diagonal(matrix[1:], [1, 2, 3, 4])

print(matrix)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

import numpy as np

# Create an 8x8 matrix filled with zeros
matrix = np.zeros((8, 8), dtype=int)

# Fill the matrix with the checkerboard pattern
matrix[1::2, ::2] = 1  # Odd rows, even columns
matrix[::2, 1::2] = 1  # Even rows, odd columns

print(matrix)

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

import numpy as np

# Define the shape of the array
shape = (6, 7, 8)

# Total number of elements in the array
total_elements = np.prod(shape)

# Flat index of the 100th element (subtract 1 because indices start from 0)
flat_index = 99

# Convert the flat index to multi-dimensional index
x, y, z = np.unravel_index(flat_index, shape)

print("Index (x, y, z) of the 100th element:", (x, y, z))


# 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

import numpy as np

# Create a single tile of the checkerboard pattern
tile = np.array([[0, 1], [1, 0]])

# Use tile function to create an 8x8 matrix by tiling the single tile
checkerboard = np.tile(tile, (4, 4))

print(checkerboard)

# 22. Normalize a 5x5 random matrix (★☆☆)

import numpy as np

# Create a 5x5 random matrix
random_matrix = np.random.rand(5, 5)

# Calculate the mean and standard deviation
mean = np.mean(random_matrix)
std_dev = np.std(random_matrix)

# Normalize the matrix
normalized_matrix = (random_matrix - mean) / std_dev

print("Original Matrix:")
print(random_matrix)
print("\nNormalized Matrix:")
print(normalized_matrix)

# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

import numpy as np

# Define the custom dtype for RGBA color
rgba_dtype = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8), ('A', np.uint8)])

# Example RGBA color
color = np.array((255, 0, 0, 255), dtype=rgba_dtype)

print("RGBA Color:")
print(color)

# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

import numpy as np

# Create a 5x3 matrix
matrix1 = np.random.rand(5, 3)

# Create a 3x2 matrix
matrix2 = np.random.rand(3, 2)

# Perform matrix multiplication
result = np.dot(matrix1, matrix2)  # Or alternatively: result = matrix1 @ matrix2

print("Result of matrix multiplication:")
print(result)

# 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

import numpy as np

# Create a 1D array
arr = np.array([1, 4, 6, 9, 2, 7, 5, 10])

# Negate elements between 3 and 8
arr[(arr > 3) & (arr < 8)] *= -1

print("Modified array:")
print(arr)

# 26. What is the output of the following script? (★☆☆)
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

9
10

# 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z

'Z**Z'
'2 << Z >> 2'
'Z <- Z'
'1j*Z'
'Z/1/1'


# 28. What are the result of the following expressions? (★☆☆)
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

'nan'
'0'
'array([0.])'

# 29. How to round away from zero a float array ? (★☆☆)

import numpy as np

# Example float array
float_array = np.array([-1.5, 2.7, -3.9, 4.2])

# Round away from zero
rounded_array = np.where(float_array >= 0, np.ceil(float_array), np.floor(float_array))

print("Original float array:", float_array)
print("Rounded away from zero array:", rounded_array)


# 30. How to find common values between two arrays? (★☆☆)

import numpy as np

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

common_values = np.intersect1d(array1, array2)
print(common_values)

# 31. How to ignore all numpy warnings (not recommended)? (★☆☆)


# 32. Is the following expressions true? (★☆☆)
np.sqrt(-1) == np.emath.sqrt(-1)


# 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)


# 34. How to get all the dates corresponding to the month of July 2016? (★★☆)


# 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)


# 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)


# 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


# 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)


# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


# 40. Create a random vector of size 10 and sort it (★★☆)
# 41. How to sum a small array faster than np.sum? (★★☆)
# 42. Consider two random array A and B, check if they are equal (★★☆)
# 43. Make an array immutable (read-only) (★★☆)
# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
# 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)
# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
# 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
# 49. How to print all the values of an array? (★★☆)
# 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
# 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
# 54. How to read the following file? (★★☆)
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
# 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
# 56. Generate a generic 2D Gaussian-like array (★★☆)
# 57. How to randomly place p elements in a 2D array? (★★☆)
# 58. Subtract the mean of each row of a matrix (★★☆)
# 59. How to sort an array by the nth column? (★★☆)
# 60. How to tell if a given 2D array has null columns? (★★☆)
# 61. Find the nearest value from a given value in an array (★★☆)
# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
# 63. Create an array class that has a name attribute (★★☆)
# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
# 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
# 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
# 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
# 69. How to get the diagonal of a dot product? (★★★)
# 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
# 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
# 72. How to swap two rows of an array? (★★★)
# 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)
# 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
# 75. How to compute averages using a sliding window over an array? (★★★)
# 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
# 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
# 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
# 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
# 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
# 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
# 82. Compute a matrix rank (★★★)
# 83. How to find the most frequent value in an array?
# 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
# 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
# 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
# 88. How to implement the Game of Life using numpy arrays? (★★★)
# 89. How to get the n largest values of an array (★★★)
# 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
# 91. How to create a record array from a regular array? (★★★)
# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
# 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
# 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
# 95. Convert a vector of ints into a matrix binary representation (★★★)
# 96. Given a two dimensional array, how to extract unique rows? (★★★)
# 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
