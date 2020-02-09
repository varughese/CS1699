<div>

## CS 1699: Homework 1

**Due:** 1/23/2020, 11:59pm

*   This assignment is worth 50 points. Each question/micro-exercise is worth 2.5 points.
*   We have starter code <span class="fixed-width">hw1.py</span> in the zip file <span class="fixed-width">hw1_starter.zip</span>, provided on CourseWeb. Your task is to complete the functions in the starter file. The specific function that you need to complete is listed in the brackets for each question. You may also need to write your answers (see below) in a file <span class="fixed-width">answers.txt</span>.
*   It is fair game to look up the Python documentation, or to look for answers on the web, assuming you look for individual functions that accomplish what you are asked, rather than entire code blocks.
*   Please use Python3 for all assignments (Python3.5+ recommended). You can also use <span class="fixed-width">numpy/scipy</span>, <span class="fixed-width">scikit-image</span> and <span class="fixed-width">matplotlib</span> libraries for this assignment.

Matrices and functions:

1.  Generate a 1000000x1 (one million by one) vector of random numbers from a Gaussian (normal) distribution with mean of 0 and standard deviation of 5. (<span class="fixed-width">generate_random_numbers</span>)
2.  Add 1 to every value in the previous list, by using a loop. To determine how many times to loop, use the <span class="fixed-width">size</span> or <span class="fixed-width">shape</span> functions. Time this operation and print the number in the code. Write that number down in <span class="fixed-width">answers.txt</span>. (<span class="fixed-width">add_one_by_loop</span>, <span class="fixed-width">measure_time_consumptions</span>)
3.  Now add 1 to every value in the original random vector, without using a loop. Time this operation, print the time and write it down. (<span class="fixed-width">add_one_without_loop</span>, <span class="fixed-width">measure_time_consumptions</span>)
4.  Plot the exponential function <span class="fixed-width">2**x</span>, for non-negative _even_ values of x smaller than 30, without using loops. Saving the figure into a file called <span class="fixed-width">exponential.png</span> for submission. (<span class="fixed-width">plot_without_loop</span>)
5.  Create a script that prints all the values between 1 and 10, in random order, with pauses of 1 second between each two prints. (<span class="fixed-width">print_one_to_ten_in_random_order_with_pauses</span>)
6.  Generate two random matrices <span class="fixed-width">A</span> and <span class="fixed-width">B</span>, and compute their product by hand, using loops. It is guaranteed that the two matrices could be multiplied. Your code should generate the same results as Python's <span class="fixed-width">A@B</span> operation or numpy's <span class="fixed-width">np.matmul()</span>. (<span class="fixed-width">matrix_multiplication_by_loop</span>)
7.  Generate a matrix of shape [10, 10] containing numbers from 0 to 99 by manipulation of a given vector. Specifically, given a vector containing numbers ranging from 0 to 9, you need to perform some matrix manipulations on the vector (addition, transpose, broadcast, etc.), and generate a matrix containing 0 to 99\. You should not initialze the desired matrix manually. (<span class="fixed-width">matrix_manipulation</span>)
8.  Write a function <span class="fixed-width">normalize_rows</span> which uses a single command (one line and no loops) to make the sum in each row of the matrix 1. More specifically, _row-wise normalization_ requires the following property to hold:
    1.  Sum of the entries in _each_ row should be 1.
    2.  If the elements in a row were not identical before the normalization, they should remain different after your normalization; however, the relative order should be preserved.Assume the input matrix to your function is (1) non-negative and (2) all rows contain at least 1 non-zero element. (<span class="fixed-width">normalize_rows</span>)
9.  Create a recursive function that returns the n-th number (n >= 1) in the Fibonacci sequence 1, 1, 2, 3, 5, 8, 13... Call it to demonstrate how it works. (<span class="fixed-width">recursive_fibonacci</span>)
10.  Implement a function that takes in a matrix <span class="fixed-width">M</span>, removes duplicate rows from that input matrix and outputs the result as matrix <span class="fixed-width">N</span>. You cannot call numpy's <span class="fixed-width">np.unique</span> or Python's <span class="fixed-width">unique</span> functions. (<span class="fixed-width">unique_rows</span>)

Images:

1.  Read <span class="fixed-width">pittsburgh.png</span> into Python as a matrix, and write down its dimensions. (<span class="fixed-width">read_image</span>)
2.  Convert the image to grayscale. There are a few different libraries for handling images in Python (such as Scikit-Image, PIL, OpenCV, etc.). The input and output of your function should be <span class="fixed-width">np.ndarray</span>, so please make sure you are dealing with the correct data type if you want to use an external library. You are also welcome to implement this function by yourself (via matrix manipulation). (<span class="fixed-width">convert_image_into_grayscale</span>)
3.  Find the darkest pixel in the image, and write its value and [row, column] in your answer file. (<span class="fixed-width">find_darkest_pixel</span>)
4.  Place a 31x31 square (a square with side equal to 31 pixels) centered on the darkest pixel from the previous question. In other words, replace all pixels in that square with white pixels. (<span class="fixed-width">mask_image_around_darkest_pixel</span>)
5.  Display the modified image (which includes the original image with a white square over it), and save the new figure to a file <span class="fixed-width">masked_image.png</span>. (<span class="fixed-width">save_image</span>)
6.  Using the original <span class="fixed-width">pittsburgh.png</span> image, compute the scalar average pixel value along each channel (R, G, B) separately, then subtract the average value per channel. Display the resulting image and write it to a file <span class="fixed-width">mean_subtracted.png</span>. (<span class="fixed-width">subtract_per_channel_mean</span>)

Text:

1.  Read in <span class="fixed-width">waugh.txt</span> as a string. Measure and write down the number of characters in it. (<span class="fixed-width">read_and_count_text</span>)
2.  Convert all text to lowercase, remove punctuation, and extract all words. Place them in an array. (<span class="fixed-width">preprocess_text</span>)
3.  Compute the frequency of each word in the text. Report the frequencies of the top-5 most frequent words in your answers file, using the format <span class="fixed-width">word1: count1, word2:count2, ...</span> (<span class="fixed-width">measure_word_frequency</span>)
4.  Use the Python built-in function <span class="fixed-width">shuffle</span> to shuffle the letters in each word, then put the words back together in a single string, and save the string to a new text file <span class="fixed-width">waugh_shuffled.txt</span>. (<span class="fixed-width">shuffle_texts_in_file</span>)

**Submission:** Please include the following files in your submission zip:

*   A completed Python file <span class="fixed-width">hw1.py</span>
*   An answers file (where answers are requested above) <span class="fixed-width">answers.txt</span>
*   Image files <span class="fixed-width">masked_image.png</span>, <span class="fixed-width">mean_subtracted.png</span> and <span class="fixed-width">exponential.png</span>
*   Text file <span class="fixed-width">waugh_shuffled.txt</span>

</div>