import collections
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.color

# You should not use any other libraries.


def generate_random_numbers(num_rows=1000000, num_cols=1, mean=0.0, std=5.0):
  """Generates random numbers using `numpy` library.

  Generates a vector of shape 1000000 x 1 (one million by one) with random
  numbers from Gaussian (normal) distribution, with mean of 0 and standard
  deviation of 5.

  Note: You can use `num_rows`, `num_cols`, `mean` and `std` directly so no need
  to hard-code these values.

  Hint: This can be done in one line of code.

  Args:
    num_rows: Optional, number of rows in the matrix. Default to be 1000000.
    num_cols: Optional, number of columns in the matrix. Default to be 1.
    mean: Optional, mean of the Gaussian distribution. Default to be 0.
    std: Optional, standard deviation of the Gaussian dist. Default to be 5.

  Returns:
    ret: A np.ndarray object containing the desired random numbers.
  """
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def add_one_by_loop(matrix):
  """Adds one to all elements in a given np.ndarray *using* loop.

  Hints:
  - The following link may be helpful for determining how many times to loop:
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html

  Args:
    matrix: A np.ndarray of shape [M, N] with arbitrary values. M represents
      number of rows in the matrix, and N represents number of columns.

  Returns:
    ret: A np.ndarray of the same shape as `matrix`, with each element being
      added by 1.
  """
  ret = matrix.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def add_one_without_loop(matrix):
  """Adds one to all elements in a given np.ndarray *without* using loop.

  Args:
    matrix: A np.ndarray of shape [M, N] with arbitrary values.

  Returns:
    ret: A np.ndarray of the same shape as `matrix`.
  """
  ret = matrix.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def measure_time_consumptions():
  """Measures the time for executing functions.

  Measures the time consumption for `add_one_by_loop` and `add_one_without_loop`
  after you completed the two functions above. You can create a random matrix
  with the function `generate_random_numbers` as input, assuming you have
  completed that function.
  Please remember to print the execution time in your code, and write down the
  numbers in answers.txt as well.

  Hint:
  - Python has built-in libararies `time` that you may find helpful:
  https://docs.python.org/3/library/time.html

  Args:
    No argument is required.

  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return None


def plot_without_loop(saving_path="exponential.png"):
  """Plots in Python3 with `matplotlib` library.

  Plot the exponential function 2**x, for non-negative even values of x smaller
  than 100, *without* using loops.

  Args:
    saving_path: Optional, path for saving the plot.

  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return None


def print_one_to_ten_in_random_order_with_pauses():
  """Prints all integers from 1 to 10 randomly with pauses.

  Prints all the values between 1 and 10, in random order, with pauses of one
  second between each two prints.

  Note: The range is inclusive on both ends (i.e. 10 numbers in total).

  Args:
    No argument is required.

  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return None


def matrix_multiplication_by_loop(matrix_a, matrix_b):
  """Calculates the matrix multiplication *using* loop.

  Given two matrices `matrix_a` and `matrix_b`, calculates the product of the
  two matrices using loop. You are *NOT* allowed to use the built-in matrix
  multiplication in this question, such as `np.matmul`, `np.dot` or `@`. You
  should implement the function using loops.

  Args:
    matrix_a: A np.ndarray of shape [M, N] with arbitrary values.
    matrix_b: A np.ndarray of shape [N, K] with arbitrary values.

  Returns:
    ret: A np.ndarray of shape [M. K] which is equivalent to the product of
      `matrix_a` by `matrix_b`.
  """
  assert matrix_a.shape[1] == matrix_b.shape[0]
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.

  # The following code is to verify that your implementation is correct.
  assert np.all(np.isclose(ret, matrix_a @ matrix_b))

  return ret


def matrix_manpulation():
  """Generates a matrix of shape [10, 10] with elements ranging from 0 to 99.

  In this question you need to take practice of manipulating numpy matrix. More
  specifically, given a np.ndarray of shape [10, 1] with elements from 0 to 9,
  you need to manipuate the vector to obtain a matrix of shape [10, 10] and
  contains elements from 0 to 99. You *cannot* manually create this new matrix,
  instead, the matrix should come from the given vector via matrix manipulation
  (addition, broadcast, transpose, etc.).

  Hints: You may find the following numpy functions useful:
  - https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html
  - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html

  Args:
    No argument is required.

  Returns:
    ret: A np.ndarray matrix of shape [10, 10] containing elements from 0 to 99.
  """
  vector = np.expand_dims(np.arange(10), 1)
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.

  return ret


def normalize_rows(matrix):
  """Normalizes the given matrix in a row-wise manner *without* using loops.

  By row-wise normalizing a matrix, the sum of the entries in each row should be
  1. If the elements in a row were not identical before the normalization, they
  should not be identical after the normalization. The relative order should be
  preserved though.

  Note: Assume that all elements in the matrix are *non-negative*, and all rows
  contain *at least* one non-zero element.

  Hint: This can be done in one line of code.

  Args:
    matrix: A np.ndarray of shape [M, N] with non-negative values.

  Returns:
    ret: A np.ndarray of the same shape as `matrix`.
  """
  assert np.all(matrix >= 0) and np.all(matrix.sum(axis=1) > 0)
  ret = matrix.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def recursive_fibonacci(n):
  """Calculates the n-th element in Fibonacci sequence.

  Note: Fibonacci sequence: https://en.wikipedia.org/wiki/Fibonacci_number

  Args:
    n: An integer greater than 0.

  Returns:
    ret: An integer representing the n-th number in Fibonacci sequence.
  """
  assert n > 0
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def my_unique(matrix):
  """Removes duplicate rows in a given matrix.

  Note: You are *not* allowed to use `np.unique` for this question. You need to
  implement this function on your own.

  Args:
    matrix: A np.ndarray of shape [M, N] with arbitrary values. There may exist
      duplicate rows in the matrix.

  Returns:
    ret: A np.ndarray of shape [M', N] where M' <= M, without duplicate rows.
  """
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def read_image(image_path="pittsburgh.png"):
  """Loads an image using `matplotlib` library.

  In this question you need to load an image using `matplotlib` library and 
  the image will be represented as a np.ndarray. Please print the dimensions of
  the image in your code, and write it down in the answers.txt.

  Args:
    image_path: Optional, location of the image to be loaded.

  Returns:
    ret: An np.ndarray of shape [M, N, 3] representing an image in RGB.
  """
  ret = None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def convert_image_to_grayscale(image):
  """Converts a RGB image into grayscale.

  Given a RGB image represented as np.ndarray, this function will convert the
  image into grayscale.

  Args:
    image: An np.ndarray of shape [M, N, 3] representing an image in RGB.

  Returns:
    ret: An np.ndarray of shape [M, N] in grayscale of the original image.
  """
  assert len(image.shape) == 3 and image.shape[2] == 3
  ret = image.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return ret


def find_darkest_pixel(grayscale_image):
  """Given a grayscale image, finds the darkest pixel in the image.

  Given a grayscale image, this function will find the darkest pixel in the
  image including its value and coordinate (row_index, column_index).

  Hints:
  - You may find the following functions helpful:
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html

  Args:
    grayscale_image: An np.ndarray of shape [M, N] representing an grayscale
      image.

  Returns:
    value: The value of the darkest pixel in the image.
    row_index: The row_index of the darkest pixel.
    column_index: The column_index of the darkest pixel.
  """
  assert len(grayscale_image.shape) == 2
  value, row_index, column_index = None, None, None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return value, (row_index, column_index)


def mask_image_around_darkest_pixel(image, patch_size=31):
  """Given a regular color image, convert it to grayscale, find the darkest
  pixel in the image and place a square centered on the darkest pixel.

  Note: You can use the functions you have implemented in previous questions.
  
  Hint:
  - By placing a square mask, you can simply replacing all pixels in that square
  with white pixels.

  Args:
    image: An np.ndarray of shape [M, N, 3] representing an image in RGB.
    patch_size: Optional, the size of the square patch. This number represents 
      the length of each side.

  Returns:
    image: An np.ndarray of shape [M, N, 3] representing an image in RGB.
  """
  masked_image = image.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return masked_image


def save_image(image, saving_path="masked_image.png"):
  """
  Saves an image represented as np.ndarray on disk.

  Args:
    image: An np.ndarray of shape [M, N, 3] representing an image in RGB.
    saving_path: Optional, the path to save the modified image.
  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return None


def subtract_per_channel_mean(image, saving_path="mean_subtracted.png"):
  """Subtracts per-channel-mean from a color image.

  Given an image in RGB, compute the scalar average pixel value along each
  channel (R, G, B) separately, then subtract the average value per channel.
  Save the modified image on disk.
  You should also print the average pixel values for each channel in your code.

  Note: You can use the functions you have implemented in previous questions.

  Hints:
  - The mean-subtracted np.ndarray may not be saved directly as image so you may
  need to normalize the values to 0 and 1.

  Args:
    image: An np.ndarray of shape [M, N, 3] representing an image in RGB.
    saving_path: Optional, the path to save the modified image.

  Returns:
    None.
  """
  image_copy = image.copy()

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return None


def read_and_count_text(text_path="waugh.txt"):
  """Loads a text file into Python and measures its length.

  In this question you need to open a text file and measures the number of
  characters in this file.

  Args:
    text_path: Optional, location of the text file to be loaded.

  Returns:
    text: A string representing the contents in the text file.
    length: An integer representing the number of characters in the text file.
  """
  text, length = None, None

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return text, length


def preprocess_text(text):
  """Preprocesses text in a given string.

  Converts all characters to lowercase, removes punctuation in the string, and
  extracts all words (assuming words are separated by space) 

  Args:
    text: A string representing texts.

  Returns:
    words: A list of string, each represents a word in the `text`.
  """
  words = []

  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return words


def measure_word_frequency(words):
  """Computes the frequency of each word from a list of words.

  Computes the frequency of each word in the list, and report the frequencies of
  the top-5 most frequent words in your answers file and print it in your code.
  Using the following format:
    word1: count1, word2:count2, ...

  Args:
    words: A list of string.

  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return words


def shuffle_texts_in_file(text_path="waugh.txt",
                          saving_path="waugh_shuffled.txt"):
  """Given a text file, shuffle the letters in each words and save to new file.

  Give a text file, read the contents into Python, preprocess the text, shuffle
  the letters in each word for all words within the text then put the words back
  together in a single string. Save the shuffled string to a new file.

  Hints:
  - You may use any functions that you have implemented in previous questions.
  - You may find the function `random.shuffle` helpful.
  https://docs.python.org/2/library/random.html#random.shuffle

  Args:
    text_path: Optional, location of the text file to be loaded.
    saving_path: Optional, the path to save the modified shuffled texts.

  Returns:
    None.
  """
  # Delete the following line and complete your implementation below.
  raise NotImplementedError
  # All your changes should be above this line.
  return words


if __name__ == "__main__":
  # Your submission should at least run successfully with the below function
  #   calls as a minimal test cases.
  # Feel free to implement more test cases to test your functions more
  #   thoroughly; we have more for grading.

  matrix = generate_random_numbers()
  measure_time_consumptions()
  plot_without_loop()
  print_one_to_ten_in_random_order_with_pauses()
  matrix_manpulation()
  normalize_rows(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
  print("The %d-th Fibonacci number is %d" % (10, recursive_fibonacci(10)))

  my_unique(matrix)

  image = read_image()
  modified = mask_image_around_darkest_pixel(image)
  save_image(modified)
  subtract_per_channel_mean(image)

  text, length = read_and_count_text()
  words = preprocess_text(text)
  measure_word_frequency(words)
  shuffle_texts_in_file()
