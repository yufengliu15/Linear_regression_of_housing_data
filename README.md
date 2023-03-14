# Linear Regression
Using the data from Lowest-rent, I created a linear regression model that would be able to predict rental prices per month, based off of user input of number of beds, baths and square footage. After the prediction, it would attempt to show 5 real world housing prices that have the same number of beds and baths, to give the user a comparison between my model and the real world. 

# Objectives
- Learn how to create a linear regression model
- Learn how to encorporate a data set into the model
- Learn how to output the data in a clean and efficient format. 

![Alt text](/terminalex.png)

Above is an example of what the user interface is like. 

# Challenges
I spent way too much time on figuring out how to implement the user interaction. This was mainly due to how I used my data. In order to make the linear regression as efficient as possible, I used a z score normalization function. If the user wished to input a number for beds for example, inputing 3 would break my model, since the data it was based off of was already normalized. Thus, I had to add the user input to X_train, and then normalize them altogether. 

Another issue was the way rentals.ca displayed their data. Some properties had ranges, such as 1-3 beds. I had to write a script that would exclude all instances of "-", which cut my dataset in half. Along with that, many property owners would put 4 beds, 2 baths and $700 for example. Here, they mean that only 1 room is available, which would inevitably skew my model. Because there is no concrete way of finding the outliers, I decided to leave it in. 

Obviously, this model does not account for location. In the future however, I do wish to attempt to find a way to encorporate location in.

# Libraries
Using numpy, matplotlib.pyplot, csv, copy and math. 
