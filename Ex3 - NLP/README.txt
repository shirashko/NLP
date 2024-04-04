############################ Natural Language Processing – Exercise 3 ############################

By: Ori Dvir - 207766478 | Shir Rashkovits - 209144013


ex3.pdf - A pdf file containing the plots of the training & validation process of all the models and the testing
results of all models (accuracy and loss on the test set and accuracy on the special sets).

ex3.py - The implementation of the exercise. To run the training and testing process of all the models run the
file ex3.py

CHANGES TO THE BASIC API:
create_or_load_slim_w2v - changed the parameter 'cache_w2v' to be True on default

train_model - changed the return value to be a tuple of 4 lists:
 - a list of the loss of the model at each epoch on the training set
 - a list of the accuracy of the model at each epoch on the training set
 - a list of the loss of the model at each epoch on the validation set
 - a list of the accuracy of the model at each epoch on the validation set

train_log_linear_with_one_hot - changed the return value to be a tuple of:
 - a pointer to the DataManager object used in the training
 - a pointer to the model object that was trained

train_log_linear_with_w2 - changed the return value to be a tuple of:
 - a pointer to the DataManager object used in the training
 - a pointer to the model object that was trained

train_lstm_with_w2v - - changed the return value to be a tuple of:
 - a pointer to the DataManager object used in the training
 - a pointer to the model object that was trained


As seen above, for all the training functions, we used the returned objects to test the models with special test
functions we created

############################ Natural Language Processing – Exercise 3 ############################
