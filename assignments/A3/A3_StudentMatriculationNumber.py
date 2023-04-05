"""
Submit a single python file with filename “A3_StudentMatriculationNumber.py”. 
Remember to rename “A3_StudentMatriculationNumber.py” using your student matriculation number,
like "A3_A1234567R.py".
(but do not rename “A3” function). 

You can only put the code in the corresponding areas and do not modify other areas.
""" 


# Only allow importing the following two packages
import numpy as np
import math # you may use math.cos(x) and math.sin(x)


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
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 1 code here.
    for i in range(num_iters):
        pass

    #>>>>>>>>>>>>>>>>>>>>

      


    # Task 2
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 2 code here.


    #>>>>>>>>>>>>>>>>>>>>




    # Task 3
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 3 code here.


    #>>>>>>>>>>>>>>>>>>>>


    # Return in this order. Do not modify it.
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 



# Do not change the following testing code.
# If you cannot run the testing code, please check your code, otherwise you will get zero mark!
############################################
learning_rate = 0.1
num_iters = 10
results = A3(learning_rate, num_iters) # your results
# Load the results of the case with learning rate of 0.1 and iteration number of 10.
groundtruth = np.load('test_case.npy') 
scores = [] 
for i in range(len(groundtruth)):
    score = 0.0
    for j in range(len(results[i])):
        if np.abs(groundtruth[i][j] - results[i][j]) < 0.001: # We bear 0.001 error in this assignment.
            score = score + (4/num_iters if i == 6 else 2/num_iters)
    scores.append(score)
# 4 full marks for f3_out and 2 for each of the others.
scores = [np.round(s, decimals=4) for s in scores]
print(f'For each output item, you get {scores} scores.')
overall_score = np.round(sum(scores), decimals=4)
print(f'In this testing case, you get {overall_score}/16.0 scores.') # Full mark is 16/16
############################################
