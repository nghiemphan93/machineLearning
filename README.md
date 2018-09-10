# Machine Learning
diverse python and machine learning algorithm examples

### 05-09-2018
Target: 
* get used to the python syntax and libraries for machine learning

Done: 
* RenameFiles: walk through all files in a directory. 
    Remove the word "Manning." in the name of each file. 
    Set new name for each file without "Manning."
* TrafficLight: Simulate color sequence of a traffic light with Class and Enum

* Notice:

### 06-09-2018
Target: 
* Learn the K Nearest Neighbor Algorithm and implement it

Problem:
* Given a set of data with label meaning which data belongs to which label
* Determine which label should be for a new random data 

Done: 
* Method classify(testData, dataSet, labels, k) and return a Tupel<label: maxVoteOfK>
* Given k => count the label of the k nearest points to the test point
* Probability of how exact the calculated result compared to the truth

Notice: 
* Took lots of time with new library stuff
* Code not worked because of different Python's version
* 2 types of Array in Python: normal and numPy's array (for direct matrix calculation)

### 07-09-2018
Target: 
* Learn Naive Bayes, Linear Regression and Neural Networks

Problem:


Done: 
* Overview of Naive Bayes and Neural Networks
* Linear regression implementation


Notice: 
* Explanation of Naive Bayes in the book "thoughtful learning with python" was unclear -> had to find some other sources to read more
* Videos are more helpful because of interactions
* Still have some troubles with plotting, visualizing data in Python -> should be mastered before getting into algorithms
* Using sklearn library first might be helpful to get some feeling about the algorithms before actually implement them 


### 10-09-2018
Target: 
* Solve linear regression with gradient descent
* Implement logistic regression with gradient descent 

Problem:
* Given a data set pairs X, y
* Find the function yHat = a*x + b to represent relationship between x and y
* In case of logistic regression: find yHat = ax1 + bx2 + c to calculate the probability of the guessed label

Done: 
* 10 iterations through each data pair, minimize error and optimize a, b and c
* Probability for logistic regression is pretty impressive, almost perfect
 

Notice: 
* Set appropriate "alpha" as learning rate so that a and b match gradually after each iteration 
* Linear Regression: already worked with the example in code but does not work for other data set -> maybe because of inappropriate "alpha"???
