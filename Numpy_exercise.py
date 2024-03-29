# 1. Import the numpy package under the name np (★☆☆)

import numpy as np

# 2. Print the numpy version and the configuration (★☆☆)

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

import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Your NumPy code here

# Reset warnings to default behavior
warnings.resetwarnings()

# 32. Is the following expressions true? (★☆☆)
np.sqrt(-1) == np.emath.sqrt(-1)

no its 'false'
# 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

from datetime import datetime, timedelta

# Get today's date
today = datetime.now().date()

# Get yesterday's date
yesterday = today - timedelta(days=1)

# Get tomorrow's date
tomorrow = today + timedelta(days=1)

print("Yesterday:", yesterday)
print("Today:", today)
print("Tomorrow:", tomorrow)

# 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

from datetime import datetime, timedelta

# Define the start date and end date of July 2016
start_date = datetime(2016, 7, 1)
end_date = datetime(2016, 8, 1)

# Generate all dates for July 2016
dates_in_july_2016 = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]

# Print all the dates
for date in dates_in_july_2016:
    print(date.date())
 
# 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

import numpy as np

# Assuming A and B are NumPy arrays
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Compute ((A+B)*(-A/2)) in place
np.add(A, B, out=A)
np.negative(A, out=A)
np.divide(A, 2, out=A)

# 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)

import numpy as np

# Generate random array of positive numbers
random_array = np.random.rand(5) * 10  # Example array

# Method : Using map and int
integer_part_method = list(map(int, random_array))


# 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

import numpy as np

# Create the matrix
matrix = np.zeros((5, 5))

# Assign row values ranging from 0 to 4
for i in range(5):
    matrix[i] = np.arange(5)

print(matrix)

# 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

import numpy as np

# Define a generator function to generate 10 integers
def generate_integers():
    for i in range(10):
        yield i

# Use the generator function to build an array
array_from_generator = np.array(list(generate_integers()))
print(array_from_generator)

# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

import numpy as np

# Create a vector of size 10 with values ranging from 0 to 1 (both excluded)
vector = np.linspace(0, 1, 12)[1:-1]

print(vector)

# 40. Create a random vector of size 10 and sort it (★★☆)

import numpy as np

# Create a random vector of size 10
random_vector = np.random.rand(10)

# Sort the vector
sorted_vector = np.sort(random_vector)

print(sorted_vector)

# 41. How to sum a small array faster than np.sum? (★★☆)

import numpy as np

# Create a small array
small_array = np.array([1, 2, 3, 4, 5])

# Sum using np.sum()
sum_np = np.sum(small_array)

# Sum using built-in sum() function
sum_builtin = sum(small_array)

print("Sum using np.sum():", sum_np)
print("Sum using built-in sum():", sum_builtin)

# 42. Consider two random array A and B, check if they are equal (★★☆)

import numpy as np

# Generate two random arrays A and B
A = np.random.rand(5)
B = np.random.rand(5)

# Check if the arrays are equal
are_equal = np.array_equal(A, B)

print("Arrays A and B are equal:", are_equal)

# 43. Make an array immutable (read-only) (★★☆)

import numpy as np

# Create a NumPy array
array = np.array([1, 2, 3, 4, 5])

# Make the array immutable
array.flags.writeable = False

# Try to modify the array (will raise an error)
try:
    array[0] = 10
except ValueError as e:
    print("Error:", e)
 
# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

import numpy as np

# Generate a random 10x2 matrix of cartesian coordinates
cartesian_coords = np.random.rand(10, 2)

# Extract x and y coordinates
x = cartesian_coords[:, 0]
y = cartesian_coords[:, 1]

# Convert cartesian coordinates to polar coordinates
r = np.linalg.norm(cartesian_coords, axis=1)
theta = np.arctan2(y, x)

# Combine r and theta to get polar coordinates
polar_coords = np.column_stack((r, theta))

print("Cartesian Coordinates:")
print(cartesian_coords)
print("\nPolar Coordinates:")
print(polar_coords)

# 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆

import numpy as np

# Create a random vector of size 10
random_vector = np.random.rand(10)

# Find the index of the maximum value
max_index = np.argmax(random_vector)

# Replace the maximum value with 0
random_vector[max_index] = 0

print("Random vector with maximum value replaced by 0:")
print(random_vector)

# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)

import numpy as np

# Define the number of points along each axis
n_points = 5

# Generate x and y coordinates covering the [0,1]x[0,1] area
x = np.linspace(0, 1, n_points)
y = np.linspace(0, 1, n_points)

# Create a meshgrid from x and y coordinates
X, Y = np.meshgrid(x, y)

# Create a structured array with x and y coordinates
structured_array = np.empty(X.shape, dtype=[('x', float), ('y', float)])
structured_array['x'] = X
structured_array['y'] = Y

print(structured_array)

# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

import numpy as np

# Example arrays X and Y
X = np.array([1, 2, 3])
Y = np.array([4, 5, 6])

# Construct the Cauchy matrix C
C = 1 / (X[:, np.newaxis] - Y)

print("Cauchy Matrix:")
print(C)

# 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

import numpy as np

# Integer types
integer_types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64]

print("Integer Types:")
for dtype in integer_types:
    info = np.iinfo(dtype)
    print(f"{dtype.__name__}: Min = {info.min}, Max = {info.max}")

# Floating point types
float_types = [np.float16, np.float32, np.float64]

print("\nFloating Point Types:")
for dtype in float_types:
    info = np.finfo(dtype)
    print(f"{dtype.__name__}: Min = {info.min}, Max = {info.max}")

# 49. How to print all the values of an array? (★★☆)

import numpy as np

# Example array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Print all the values of the array
print(array)
 
# 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

import numpy as np

# Example vector
vector = np.array([1, 2, 3, 4, 5])

# Given scalar
scalar = 3.5

# Find the index of the closest value to the scalar
index = np.abs(vector - scalar).argmin()

# Closest value
closest_value = vector[index]

print("Closest value to", scalar, "in the vector:", closest_value)
 
# 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

import numpy as np

# Define the structured array dtype
dtype = [('position', [('x', float), ('y', float)]),
         ('color', [('r', int), ('g', int), ('b', int)])]

# Create an empty structured array with the defined dtype
structured_array = np.empty(1, dtype=dtype)

# Fill in some example values
structured_array['position'] = (3.5, 2.0)
structured_array['color'] = (255, 0, 0)  # Red color

print(structured_array)

# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

import numpy as np

# Generate a random vector of shape (100, 2) representing coordinates
coordinates = np.random.rand(100, 2)

# Calculate point-by-point distances using broadcasting
# Reshape the coordinates array to have a new axis for broadcasting
distances = np.sqrt(np.sum((coordinates[:, np.newaxis, :] - coordinates) ** 2, axis=2))

print(distances)

# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

import numpy as np

# Create a float (32 bits) array
float_array = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)

# Convert the float array to an integer (32 bits) array in place
float_array.view(dtype=np.int32)[:] = float_array

print(float_array)

# 54. How to read the following file? (★★☆)
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11

# Open the file
with open('file.txt', 'r') as file:
    # Read lines from the file
    lines = file.readlines()

    # Initialize an empty list to store the data
    data = []

    # Process each line
    for line in lines:
        # Split the line by commas and strip whitespace from each element
        elements = [element.strip() for element in line.split(',')]

        # Convert non-empty elements to integers
        integers = [int(element) if element.strip() else None for element in elements]

        # Append the processed line to the data list
        data.append(integers)

print(data)

# 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

import numpy as np

# Example array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Enumerate over the array
for index, value in np.ndenumerate(array):
    print("Index:", index, "Value:", value)
 
# 56. Generate a generic 2D Gaussian-like array (★★☆)

import numpy as np

def gaussian_2d(shape, center=(0, 0), sigma=(1, 1)):
    """Generate a 2D Gaussian-like array."""
    x = np.arange(shape[0]) - center[0]
    y = np.arange(shape[1]) - center[1]
    xx, yy = np.meshgrid(x, y)
    exponent = -0.5 * ((xx / sigma[0]) ** 2 + (yy / sigma[1]) ** 2)
    gaussian = np.exp(exponent)
    return gaussian / np.sum(gaussian)

# Example usage
shape = (5, 5)  # Shape of the array
center = (2, 2)  # Center of the Gaussian
sigma = (1, 2)  # Standard deviations along x and y axes

gaussian_array = gaussian_2d(shape, center, sigma)
print(gaussian_array)
 
# 57. How to randomly place p elements in a 2D array? (★★☆)

import numpy as np

def randomly_place_elements(array, p, value):
    """Randomly place p elements with a specified value in a 2D array."""
    # Flatten the array to simplify indexing
    flat_array = array.flatten()
    
    # Determine the total number of elements in the array
    total_elements = array.size
    
    # Determine the number of elements to randomly place
    num_elements_to_place = min(p, total_elements)
    
    # Generate random indices to place the elements
    indices = np.random.choice(total_elements, size=num_elements_to_place, replace=False)
    
    # Set the specified value at the randomly chosen indices
    flat_array[indices] = value
    
    # Reshape the modified flat array back to the original shape
    return flat_array.reshape(array.shape)

# Example usage
array = np.zeros((5, 5))  # Example 5x5 array filled with zeros
p = 5  # Number of elements to randomly place
value = 1  # Value to place at random locations

result_array = randomly_place_elements(array, p, value)
print(result_array)
 
# 58. Subtract the mean of each row of a matrix (★★☆)

import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Calculate the mean of each row
row_means = np.mean(matrix, axis=1, keepdims=True)

# Subtract the mean of each row from the corresponding row
result_matrix = matrix - row_means

print("Result Matrix:")
print(result_matrix)

# 59. How to sort an array by the nth column? (★★☆)

import numpy as np

# Example array
array = np.array([[3, 2, 5],
                  [1, 4, 6],
                  [7, 0, 8]])

# Specify the column index to sort by
n = 1

# Get the indices that would sort the array by the nth column
sorted_indices = array[:, n].argsort()

# Sort the array by the nth column
sorted_array = array[sorted_indices]

print("Sorted Array by the", n, "th column:")
print(sorted_array)

# 60. How to tell if a given 2D array has null columns? (★★☆)

import numpy as np

def has_null_columns(array):
    """Check if a 2D array has null columns."""
    # Check if any column contains only zeros
    return np.any(np.all(array == 0, axis=0))

# Example 2D array
array = np.array([[1, 0, 3],
                  [0, 0, 0],
                  [4, 0, 6]])

if has_null_columns(array):
    print("The array has null columns.")
else:
    print("The array does not have null columns.")

# 61. Find the nearest value from a given value in an array (★★☆)

import numpy as np

def find_nearest(array, value):
    """Find the nearest value from a given value in an array."""
    nearest_index = np.abs(array - value).argmin()
    return array[nearest_index]

# Example array
array = np.array([1, 3, 5, 7, 9])

# Given value
value = 4

nearest_value = find_nearest(array, value)
print("Nearest value to", value, "in the array:", nearest_value)

# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

import numpy as np

# Example arrays
array1 = np.array([[1, 2, 3]])
array2 = np.array([[4], [5], [6]])

# Get iterators for the arrays
iter1 = np.nditer(array1)
iter2 = np.nditer(array2)

# Initialize sum
sum_result = 0

# Iterate over elements and compute sum
for element1, element2 in zip(iter1, iter2):
    sum_result += element1 + element2

print("Sum of arrays:", sum_result)

# 63. Create an array class that has a name attribute (★★☆)

import numpy as np

class NamedArray(np.ndarray):
    def __new__(cls, array, name=None):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

# Example usage
array = NamedArray([[1, 2, 3], [4, 5, 6]], name="Example Array")
print(array)
print("Name of the array:", array.name)

# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

import numpy as np

# Given vector
vector = np.array([1, 2, 3, 4, 5])

# Indices vector
indices = np.array([0, 1, 2, 1, 0])

# Get unique indices and counts
unique_indices, counts = np.unique(indices, return_counts=True)

# Add 1 to each element indexed by unique indices
vector += np.bincount(indices, minlength=len(vector))

print("Vector after adding 1 to each element indexed by the second vector:")
print(vector)

# 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

import numpy as np

# Given vector X
X = np.array([1, 2, 3, 4, 5])

# Index list I
I = np.array([0, 1, 2, 1, 0])

# Calculate the accumulated elements based on the index list
F = np.bincount(I, weights=X)

print("Accumulated elements based on the index list:")
print(F)

# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

import numpy as np

# Example image (w, h, 3) with dtype ubyte
image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

# Reshape the image to (w*h, 3) to treat each pixel as a separate color
pixels = np.reshape(image, (-1, 3))

# Convert the pixels to a list of tuples to use np.unique
pixels_list = [tuple(pixel) for pixel in pixels]

# Find the unique colors
unique_colors = np.unique(pixels_list, axis=0)

# Compute the number of unique colors
num_unique_colors = len(unique_colors)

print("Number of unique colors:", num_unique_colors)

# 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

import numpy as np

# Example four-dimensional array
array = np.random.randint(0, 10, size=(2, 3, 4, 5))

# Compute the sum over the last two axes at once
sum_over_last_two_axes = np.sum(array, axis=(-2, -1))

print("Sum over the last two axes:")
print(sum_over_last_two_axes)

# 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)

import numpy as np

# Example one-dimensional vector D
D = np.array([1, 2, 3, 4, 5, 6])

# Example vector S describing subset indices
S = np.array([0, 1, 0, 1, 1, 0])

# Find unique subset indices and their counts
unique_indices, counts = np.unique(S, return_counts=True)

# Compute means of subsets using bincount and counts
means = np.bincount(S, weights=D) / counts

print("Means of subsets:")
print(means)

# 69. How to get the diagonal of a dot product? (★★★)
import numpy as np

# Example matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Compute dot product
dot_product = np.einsum('ij,jk->ik', A, B)

# Get the diagonal of the dot product
diagonal = np.diag(dot_product)

print("Diagonal of the dot product:")
print(diagonal)


# 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

import numpy as np

# Original vector
original_vector = np.array([1, 2, 3, 4, 5])

# Number of consecutive zeros to insert between each value
consecutive_zeros = 3

# Calculate the length of the new vector
new_length = len(original_vector) + (len(original_vector) - 1) * consecutive_zeros

# Initialize the new vector with zeros
new_vector = np.zeros(new_length, dtype=original_vector.dtype)

# Assign values from the original vector to the new vector with interleaved zeros
new_vector[::consecutive_zeros+1] = original_vector

print("New vector with 3 consecutive zeros interleaved between each value:")
print(new_vector)

# 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

import numpy as np

# Example arrays
array1 = np.random.rand(5, 5, 3)
array2 = np.random.rand(5, 5)

# Multiply array1 by array2 using broadcasting
result = array1 * array2[:,:,np.newaxis]

print("Resulting array shape:", result.shape)

# 72. How to swap two rows of an array? (★★★)

import numpy as np

# Example array
array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Swap rows 1 and 2 (0-indexed)
array[1], array[2] = array[2].copy(), array[1].copy()

print("Array after swapping rows:")
print(array)

# 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)

import numpy as np

# Example set of 10 triplets describing triangles
triplets = np.array([
    [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0],
    [1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 0, 1]
])

# Initialize a set to collect unique line segments
line_segments = set()

# Iterate over each triangle
for triplet in triplets:
    # Extract line segments
    segments = [(triplet[i], triplet[(i + 1) % 3]) for i in range(3)]
    # Convert segments to tuples of sorted vertex indices to ensure uniqueness
    sorted_segments = [tuple(sorted(segment)) for segment in segments]
    # Add segments to the set
    line_segments.update(sorted_segments)

print("Set of unique line segments:")
print(line_segments)

# 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

import numpy as np

def array_from_bincount(C):
    """Produce an array A such that np.bincount(A) == C."""
    # Calculate the length of the resulting array
    length = np.sum(C)
    
    # Create an array A containing the bin indices based on the bin counts in C
    A = np.repeat(np.arange(len(C)), C)
    
    # Use cumsum to shift the elements in A to match the bin counts in C
    A[1:] = np.cumsum(A[:-1] != A[1:])
    
    return A[:length]

# Example sorted array C corresponding to a bincount
C = np.array([1, 2, 3, 4])

# Produce array A such that np.bincount(A) == C
A = array_from_bincount(C)

print("Array A such that np.bincount(A) == C:")
print(A)
 
# 75. How to compute averages using a sliding window over an array? (★★★)

import numpy as np

def sliding_window_average(array, window_size):
    """Compute averages using a sliding window over an array."""
    # Pad the array with zeros to handle edge cases
    padded_array = np.pad(array, (window_size // 2, window_size // 2), mode='edge')
    
    # Use a 1D convolution to compute the sliding window averages
    sliding_window_avg = np.convolve(padded_array, np.ones(window_size) / window_size, mode='valid')
    
    return sliding_window_avg

# Example array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Window size for sliding window
window_size = 3

# Compute averages using a sliding window over the array
averages = sliding_window_average(array, window_size)

print("Averages using a sliding window of size", window_size, ":", averages)

# 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
import numpy as np

Z = np.array([1, 2, 3, 4, 5])  # Example array

# Create a two-dimensional array with shifting rows
result = np.lib.stride_tricks.as_strided(Z, shape=(len(Z) - 2, 3), strides=(Z.strides[0], Z.strides[0]))

print(result)
[[1 2 3]
 [2 3 4]
 [3 4 5]]

# 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
x = True
x = not x
print(x)  # Output: False
x = 3.14
x *= -1
print(x) # Output: -3.14

# 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
import numpy as np

def distance_to_line(P0, P1, p):
    # Convert points to numpy arrays for easy computation
    P0 = np.array(P0)
    P1 = np.array(P1)
    p = np.array(p)

    # Calculate the vector representing the line
    v = P1 - P0

    # Calculate the vector representing the line segment from P0 to p
    w = p - P0

    # Calculate the squared length of the line segment
    w_squared_length = np.dot(w, w)

    # Calculate the parameter t which represents the projection of w onto v
    t = np.dot(w, v) / np.dot(v, v)

    # Check if the projection falls beyond the line segment
    if t < 0:
        return np.sqrt(np.dot(w, w))
    elif t > 1:
        return np.sqrt(np.dot(p - P1, p - P1))

    # Calculate the closest point on the line to the point p
    closest_point = P0 + t * v

    # Calculate the distance between the closest point and the point p
    distance = np.sqrt(np.dot(p - closest_point, p - closest_point))

    return distance

# Example usage:
P0 = [1, 1]  # Start point of the line
P1 = [4, 5]  # End point of the line
p = [2, 3]   # Point p

distance = distance_to_line(P0, P1, p)
print(distance)
    
# 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
import numpy as np

def distance_to_line(P0, P1, P):
    distances = []
    for i in range(len(P0)):
        # Convert points to numpy arrays for easy computation
        P0_i = np.array(P0[i])
        P1_i = np.array(P1[i])

        # Calculate the vector representing the line
        v = P1_i - P0_i

        for j in range(len(P)):
            # Convert point to numpy array
            p = np.array(P[j])

            # Calculate the vector representing the line segment from P0 to p
            w = p - P0_i

            # Calculate the squared length of the line segment
            w_squared_length = np.dot(w, w)

            # Calculate the parameter t which represents the projection of w onto v
            t = np.dot(w, v) / np.dot(v, v)

            # Check if the projection falls beyond the line segment
            if t < 0:
                distance = np.sqrt(np.dot(w, w))
            elif t > 1:
                distance = np.sqrt(np.dot(p - P1_i, p - P1_i))
            else:
                # Calculate the closest point on the line to the point p
                closest_point = P0_i + t * v

                # Calculate the distance between the closest point and the point p
                distance = np.sqrt(np.dot(p - closest_point, p - closest_point))

            distances.append(distance)

    return distances

# Example usage:
P0 = [[1, 1], [2, 2]]  # Start points of the lines
P1 = [[4, 5], [3, 6]]  # End points of the lines
P = [[2, 3], [5, 4], [1, 1]]  # Points to compute distance from

distances = distance_to_line(P0, P1, P)
print(distances)
        
# 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
import numpy as np

def extract_subpart(array, shape, center, fill_value=0):
    # Calculate the starting index for the subpart
    start_index = np.subtract(center, np.array(shape) // 2)
    
    # Calculate the ending index for the subpart
    end_index = np.add(start_index, shape)
    
    # Create a mask to identify the valid indices within the original array
    valid_indices = np.logical_and(start_index >= 0, end_index <= np.array(array.shape))
    
    # Initialize the subpart array with fill values
    subpart = np.full(shape, fill_value)
    
    # Calculate the valid indices within the subpart array
    valid_indices_subpart = np.logical_and(valid_indices, valid_indices.shape)
    
    # Calculate the valid indices within the original array
    valid_indices_array = np.logical_and(valid_indices, array.shape)
    
    # Extract the valid portion from the original array and place it in the subpart array
    subpart[valid_indices_subpart] = array[start_index[valid_indices_array[0]], start_index[valid_indices_array[1]]]
    
    return subpart

# Example usage:
array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

shape = (3, 3)  # Shape of the subpart
center = (1, 1)  # Center element
fill_value = 0  # Fill value for padding

subpart = extract_subpart(array, shape, center, fill_value)
print(subpart)
    
# 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
Z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

R = [Z[i:i+4] for i in range(len(Z)-3)]

print(R)
[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12, 13], [11, 12, 13, 14]]
import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(matrix)

print("Matrix rank:", rank)

# 82. Compute a matrix rank (★★★)
import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(matrix)

print("Matrix rank:", rank)

# 83. How to find the most frequent value in an array?
from collections import Counter

# Example array
array = [1, 2, 3, 4, 2, 2, 3, 2, 5, 2, 3]

# Count occurrences of each element
counter = Counter(array)

# Find the most common value
most_common_value = counter.most_common(1)[0][0]

print("Most frequent value:", most_common_value)

# 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
import numpy as np

# Create a random 10x10 matrix
matrix = np.random.randint(0, 10, size=(10, 10))

# Extract all contiguous 3x3 blocks
blocks = []
for i in range(8):  # Loop through rows
    for j in range(8):  # Loop through columns
        block = matrix[i:i+3, j:j+3]
        blocks.append(block)

# Convert the list of blocks to a numpy array
blocks_array = np.array(blocks)

print(blocks_array)
    
# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
import numpy as np

class SymmetricArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __getitem__(self, index):
        i, j = index
        if i > j:
            i, j = j, i
        return super().__getitem__((i, j))

# Example usage:
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

sym_arr = SymmetricArray(arr)

print(sym_arr[0, 1])  # Output: 2
print(sym_arr[1, 0])  # Output: 2

# 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
import numpy as np

def sum_of_matrix_products(matrices, vectors):
    # Broadcast the matrices and vectors for element-wise multiplication
    products = matrices * vectors[:, np.newaxis, :]

    # Sum along the first axis (p matrices)
    result = np.sum(products, axis=0)

    return result

# Example usage:
p = 3
n = 4

# Generate random matrices and vectors
matrices = np.random.rand(p, n, n)
vectors = np.random.rand(p, n, 1)

# Compute the sum of matrix products
result = sum_of_matrix_products(matrices, vectors)

print(result)

# 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
import numpy as np

# Create a 16x16 array (example)
array = np.random.randint(0, 10, size=(16, 16))

# Reshape the array into a 4D array with block size 4x4
blocks = array.reshape(4, 4, 4, 4)

# Sum along two axes to get the block-sum
block_sum = blocks.sum(axis=(2, 3))

print(block_sum)

# 88. How to implement the Game of Life using numpy arrays? (★★★)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(frameNum, img, grid, N):
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Compute the sum of neighbors' states using toroidal boundary conditions
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)

            # Apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = OFF
            else:
                if total == 3:
                    new_grid[i, j] = ON

    # Update the data
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img

# Game of Life parameters
ON = 255
OFF = 0
N = 100
updateInterval = 50

# Create a grid of random initial states
grid = np.random.choice([ON, OFF], N*N, p=[0.2, 0.8]).reshape(N, N)

# Set up the animation
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N),
                              frames=10,
                              interval=updateInterval,
                              save_count=50)

plt.show()
            
# 89. How to get the n largest values of an array (★★★)
import numpy as np

# Example array
array = np.array([1, 5, 2, 10, 6, 3, 8])

# Number of largest values to retrieve
n = 3

# Get indices of n largest values
indices = np.argsort(array)[-n:]

# Get n largest values
largest_values = array[indices]

print("Indices of {} largest values: {}".format(n, indices))
print("N largest values:", largest_values)

# 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
from itertools import product

# Example vectors
vector1 = [1, 2, 3]
vector2 = ['a', 'b']
vector3 = ['x', 'y', 'z']

# Build the Cartesian product
cartesian_product = list(product(vector1, vector2, vector3))

print(cartesian_product)

# 91. How to create a record array from a regular array? (★★★)
import numpy as np

# Regular array
regular_array = np.array([(1, 'John', 25),
                          (2, 'Alice', 30),
                          (3, 'Bob', 35)],
                         dtype=[('ID', int), ('Name', 'U10'), ('Age', int)])

# Convert to record array
record_array = np.rec.array(regular_array)

print(record_array)

# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
import numpy as np

# Create a large vector Z (example)
Z = np.random.randint(1, 10, size=1000000)

# Method 1: Element-wise multiplication
result1 = Z * Z * Z

# Method 2: Using np.power()
result2 = np.power(Z, 3)

# Method 3: Using the ** operator
result3 = Z ** 3

# Check if the results are the same
print(np.allclose(result1, result2) and np.allclose(result1, result3))

# 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
import numpy as np

# Example arrays A and B
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [2, 3, 1],
              [7, 8, 9],
              [3, 2, 1],
              [1, 3, 2],
              [6, 5, 4],
              [9, 8, 7]])

B = np.array([[2, 3],
              [1, 2]])

# Sort the rows of A and B
sorted_A = np.sort(A, axis=1)
sorted_B = np.sort(B, axis=1)

# Reshape B for broadcasting
sorted_B = sorted_B[:, :, np.newaxis]

# Compare rows of A with each row of B
rows_containing_elements = np.any(np.all(sorted_A[:, :, np.newaxis] == sorted_B, axis=1), axis=1)

# Get the rows of A that contain elements of each row of B
result = A[rows_containing_elements]

print(result)

# 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
import numpy as np

# Example 10x3 matrix
matrix = np.array([[1, 2, 3],
                   [2, 2, 2],
                   [4, 5, 6],
                   [7, 7, 7],
                   [2, 2, 3],
                   [8, 8, 8],
                   [9, 9, 9],
                   [2, 2, 2],
                   [5, 5, 5],
                   [1, 2, 1]])

# Extract rows with unequal values
unequal_rows = matrix[np.any(matrix[:, 1:] != matrix[:, :-1], axis=1)]

print(unequal_rows)

# 95. Convert a vector of ints into a matrix binary representation (★★★)
import numpy as np

# Example vector of integers
vector = np.array([2, 5, 7])

# Compute the maximum value in the vector to determine the number of bits needed
max_value = np.max(vector)

# Calculate the number of bits needed to represent the maximum value
num_bits = int(np.ceil(np.log2(max_value + 1)))

# Initialize an empty matrix to store the binary representation
matrix = np.zeros((len(vector), num_bits), dtype=int)

# Convert each integer to its binary representation
for i, num in enumerate(vector):
    binary_repr = np.binary_repr(num, width=num_bits)
    matrix[i] = [int(bit) for bit in binary_repr]

print(matrix)

# 96. Given a two dimensional array, how to extract unique rows? (★★★)
import numpy as np

# Example two-dimensional array
array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [1, 2, 3],
                  [7, 8, 9],
                  [4, 5, 6]])

# Extract unique rows
unique_rows = np.unique(array, axis=0)

print(unique_rows)

# 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
inner_product = np.einsum('i,i->', A, B)
outer_product = np.einsum('i,j->ij', A, B)
sum_result = np.einsum('i->', A)  # Sum of elements of vector A
elementwise_mul = np.einsum('i,i->i', A, B)

# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
import numpy as np

# Example vectors (X, Y) describing the path
X = np.array([0, 1, 3, 5, 6])
Y = np.array([0, 2, 4, 3, 1])

# Calculate the total path length
path_length = np.sum(np.sqrt(np.diff(X)**2 + np.diff(Y)**2))

# Define the number of equidistant samples
num_samples = 10

# Calculate the distance between equidistant samples
step_size = path_length / (num_samples - 1)

# Interpolate to generate equidistant samples
samples_X = np.interp(np.arange(0, path_length, step_size), np.cumsum(np.sqrt(np.diff(X)**2 + np.diff(Y)**2)), X)
samples_Y = np.interp(np.arange(0, path_length, step_size), np.cumsum(np.sqrt(np.diff(X)**2 + np.diff(Y)**2)), Y)

# Combine X and Y coordinates of equidistant samples
equidistant_samples = np.column_stack((samples_X, samples_Y))

print(equidistant_samples)

# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
import numpy as np

def is_multinomial_row(row, n):
    # Check if the row contains only integers and if the sum equals n
    return np.all(np.equal(np.mod(row, 1), 0)) and np.sum(row) == n

def select_multinomial_rows(X, n):
    # Select rows from X that satisfy the condition
    return X[np.array([is_multinomial_row(row, n) for row in X])]

# Example 2D array X
X = np.array([[1, 2, 3],
              [2, 3, 4],
              [1, 1.5, 2.5],
              [4, 5, 6]])

# Example value of n
n = 6

# Select rows from X that can be interpreted as draws from a multinomial distribution with n degrees
selected_rows = select_multinomial_rows(X, n)

print(selected_rows)

# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

import numpy as np

def bootstrap_confidence_interval(X, N, alpha=0.05):
    # Generate N bootstrap samples with replacement
    bootstrap_samples = np.random.choice(X, size=(N, len(X)), replace=True)

    # Compute the mean of each bootstrap sample
    means = np.mean(bootstrap_samples, axis=1)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    # Compute the confidence interval
    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)

    return lower_bound, upper_bound

# Example 1D array X
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Number of bootstrap samples
N = 1000

# Compute the bootstrapped 95% confidence interval for the mean of X
lower_bound, upper_bound = bootstrap_confidence_interval(X, N)

print("Bootstrapped 95% confidence interval for the mean of X:", (lower_bound, upper_bound))
    
