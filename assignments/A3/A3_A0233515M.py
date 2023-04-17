"""
Submit a single python file with filename “A3_StudentMatriculationNumber.py”. 
Remember to rename “A3_StudentMatriculationNumber.py” using your student matriculation number,
like "A3_A1234567R.py".
(but do not rename “A3” function). 

You can only put the code in the corresponding areas and do not modify other areas.
"""


# Only allow importing the following two packages
import numpy as np
import math  # you may use math.cos(x) and math.sin(x)

# do not modify the function name in this assignment, just let it as "A3"


def A3(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """

    # Task 1
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    # <<<<<<<<<<<<<<<<<<<<
    # Put your task 1 code here.

    # Gradient descent

    a = 1  # Initialization of a = 1.0

    for i in range(0, num_iters):
        # Gradient of cost function a**6 by differentiation
        gradient_of_cost_function = 6*(a**5)
        a = a - learning_rate * gradient_of_cost_function  # Update a
        # Replace value in the corresponding iteration position of a_out with the value of a in the current iteration
        a_out[i] = a
        # Replace value in the corresponding iteration position of f1_out with the value of a in the current iteration
        f1_out[i] = a**6

    # >>>>>>>>>>>>>>>>>>>>

    # Task 2
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    # <<<<<<<<<<<<<<<<<<<<
    # Put your task 2 code here.

    # Gradient descent

    b = 1.2  # Initialization of b = 1.2

    for i in range(0, num_iters):
        # Gradient of cost function cos^2(b) by differentiation math.cos/sin inputs are in radians
        gradient_of_cost_function = -2*(math.cos(b))*(math.sin(b))
        b = b - learning_rate * gradient_of_cost_function  # Update b
        # Replace value in the corresponding iteration position of b_out with the value of b in the current iteration
        b_out[i] = b
        # Replace value in the corresponding iteration position of f2_out with the value of b in the current iteration
        f2_out[i] = (math.cos(b))**2

    # >>>>>>>>>>>>>>>>>>>>

    # Task 3
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    # <<<<<<<<<<<<<<<<<<<<
    # Put your task 3 code here.

    c = 0.5  # Initialization of c = 0.5
    d = 2.5  # Initialization of d = 2.5

    for i in range(0, num_iters):
        # Gradient of cost function for c (partial differentiation)
        gradient_of_cost_function_c = 2 * (c + 2 * d - 7) + 4 * (2 * c + d - 5)
        # Gradient of cost function for d (partial differentiation)
        gradient_of_cost_function_d = 4 * (c + 2 * d - 7) + 2 * (2 * c + d - 5)

        c = c - learning_rate * gradient_of_cost_function_c  # Update c
        d = d - learning_rate * gradient_of_cost_function_d  # Update d

        # Replace value in the corresponding iteration position of c_out with the value of c in the current iteration
        c_out[i] = c
        # Replace value in the corresponding iteration position of d_out with the value of d in the current iteration
        d_out[i] = d
        # Replace value in the corresponding iteration position of f3_out with the value of cost function (c, d) in the current iteration
        f3_out[i] = (c + 2 * d - 7) ** 2 + (2 * c + d - 5) ** 2

    # >>>>>>>>>>>>>>>>>>>>

    # Return in this order. Do not modify it.
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out


# Do not change the following testing code.
# If you cannot run the testing code, please check your code, otherwise you will get zero mark!
############################################
learning_rate = 0.1
num_iters = 10
results = A3(learning_rate, num_iters)  # your results
# Load the results of the case with learning rate of 0.1 and iteration number of 10.
groundtruth = np.load('test_case.npy')
scores = []
for i in range(len(groundtruth)):
    score = 0.0
    for j in range(len(results[i])):
        # We bear 0.001 error in this assignment.
        if np.abs(groundtruth[i][j] - results[i][j]) < 0.001:
            score = score + (4/num_iters if i == 6 else 2/num_iters)
    scores.append(score)
# 4 full marks for f3_out and 2 for each of the others.
scores = [np.round(s, decimals=4) for s in scores]
print(f'For each output item, you get {scores} scores.')
overall_score = np.round(sum(scores), decimals=4)
# Full mark is 16/16
print(f'In this testing case, you get {overall_score}/16.0 scores.')
############################################
