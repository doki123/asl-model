A neural network used to identify American Sign Language letters and numbers

**USING THE KAGGLE MODEL:**

1. Create a new notebook in Google Colab, and mount the Google Drive.
2. Download the dataset from https://www.kaggle.com/code/gpiosenka/callback-continue-or-halt-training-f1-96/notebook into Google Drive. 
	- Originally, the entire dataset was downloaded directly into the user's computer system. However, due to the size of the complete dataset, many classes did not end up downloaded. Next, subfolders were downloaded individually into Google Drive to ensure all images get transferred; while downloading the different classes, however, the disparity of image counts between classes was becoming increasingly large (eg the sign for _0_ had barely 100 images, while the sign for _a_ had over 1000). In order to both reduce the difference and the download time for the dataset, the subfolders were manually cancelled once 500-1000 images were downloaded. 
3. At line 35 in kaggle_model.py, line _sdir = 'drive/MyDrive/processed_combine_asl_dataset'_, set the directory to the location of the dataset in the Drive.   
4. At line 129, line _max_samples = 50_, the original source code used 250 samples. However, at 50 samples, the accuracy of the model was still in the 90s while reducing the training time significantly. Additionally, on the next line, while _min_samples = 0_ does work well for the model without incident, it can still be set higher should the user wish so.
5. At line 217, line _working_dir = 'drive/MyDrive'_, the directory must once again be set to the Google Drive so that augmented images can be stored within. 
6. From lines 311-371, a function is created wherein after a certain amount of epochs, the model training will stop and ask for input from the user for a) whether the user wants to continue training and b), should they want to continue training, how many epochs the user wants trained until the computer asks for input again. 
	- In the next section, at lines 374 and 375, variable _epochs_ is the maximum amount of epochs that the machine will potentiall run in total. Variable _ask_epoch_ is how many epochs the machine will run before asking for input. If you want to avoid repeatedly checking your computer and giving inputs for your desired epoch length, set _ask_epochs_ to the full epochs you want ran. This way, you will only have to give input once to end the training. 
7. From lines 439-479, a function _def predictor()_ is created specifically to predict on the testing dataset and create a confusion matrix that displays the accuracy of the model. However, a simple model.predict() can be used to predict on single images, which can be seen in the companion file. 
8. At line 486, the model is saved in a file named after the accuracy of the model; however, it can also be more simply named to whatever you want by deleting the above four lines and simply writing _save_id = '[INSERT_TITLE].h5_.


**USING THE WEBCAM AND PREDICTIONS**



This application is a neural network created in Google Colab, based on source code and trained on a dataset sourced from Kaggle. Working in tandem to this, images from a webcam are captured and processed using the mediapipe library. This converted data can then be fed into a saved version of the model in order to return its predicted classification. Mediapipe can superimpose a hand’s major points–specifically, the joints of fingers and the palm–onto an image and return the coordinates of each landmark. The usage of this library can increase the accuracy of the predictions by cutting out the background noise of colors and obscuring objects. 


Additionally, though Google Colab was the chosen platform for this project specifically due to its non-local location, this caused difficulty accessing the computer’s local webcam to capture images. This created a secondary issue wherein mediapipe could not be directly applied onto live footage, in contrast to a prototype model that captured and converted images into landmarks in real time. However, a workaround was ultimately implemented where separate images are captured and then processed afterward. 

There were difficulties feeding in self-captured images into the model itself during the training process, which is being worked on by pre-training the model, saving it, and using self-captured images only for predictions.

While testing the model, it was observed that predictions were consistently skewed to the ‘f’ or ‘g’ sign, despite the self-calculated accuracy for the model being in the high 90s. While combing through the model and comparing the outputs to the original Kaggle code, it was noticed that there were far fewer files in the dataset as compared to the original code; furthermore, there were only six classes in total noted in the dataframe, while there should have been 36. 

Further examination led to the conclusion that in the process of downloading the images to a google drive for use, many of the subfolders had been dropped due to the size of the dataset. To rectify this, the dataset was redownloaded subfolder by subfolder, but that too proved difficult due to the long wait-time of downloads, should the dataset be downloaded in its entirety. However, upon closer inspection, it was noticed that some classes of signs only had 60 or so images, while others had images into the thousands. In order to both cut down on the wait-time and to equalize the large disparity in data, the downloads were manually halted once around five hundred to one thousand images had been downloaded into each folder. 

Features I would like to add is a full, working HTML website from which to access this model and predict signs. I would also like to supplement more data into each of the classes to further equalize the data quantities in each class.  

In order to recreate this model, first, the dataset (linked below) will need to be downloaded into a google drive. This google drive will then need to be mounted to a new google colab notebook in order for the model to train on the images. The amount of epochs that are run can depend on the user–decent accuracy was achieved after the first five alone, but more can be run if so desired. Additionally, epochs will be around 6 to 8 minutes per epoch if the full dataset is used; however, the dataset samples used for training can be reduced to 20% if the user desires a faster training time. Even with a reduced training dataset, accuracy values in the 80 and 90 percent ranges are reached. Finally, in order to access the saved model afterwards, save the model into the mounted google drive. 

In order to run the webcam portion of the code, the user will first need to create a folder to save captured images in. In google colab, a new folder can be automatically created with the !mkdir new_folder command, or the user can alternatively store the images inside a mounted google drive. The user may also need to run a !pip install mediapipe command should the mediapipe sections of code not function correctly. Save the captured images within the designated folder and run the mediapipe section of code in order to capture the hand landmarks. 

The dataset and source code for the model come from this Kaggle post: https://www.kaggle.com/code/gpiosenka/callback-continue-or-halt-training-f1-96/notebook. 
