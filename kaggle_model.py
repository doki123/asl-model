# IMPORT NECESSARY MODULES

import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2

sns.set_style('darkgrid')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
# prevent annoying tensorflow warning

# READ IN IMAGES AND CREATE A DATAFRAME OF IMAGE PATHS AND CLASS LABELS

sdir = 'drive/MyDrive/processed_combine_asl_dataset'  
classlist = sorted(os.listdir(sdir))  # creates an alphabetical list of classes taken from sub-folder names 
print(classlist)
filepaths = []
labels = [] 

for klass in classlist:  # runs through list of classes
    classpath = os.path.join(sdir, klass)  # retrieves sub-folder routes 
    flist = sorted(os.listdir(classpath))  

    for f in flist:  # runs through list of subfolder routes
        fpath = os.path.join(classpath,f)  # retrieves individual image routes
        filepaths.append(fpath)
        labels.append(klass)

# A Series is a 1-Dimensional array with axis labels
Fseries = pd.Series(filepaths, name='filepaths')  
Lseries = pd.Series(labels, name='labels')  
df = pd.concat([Fseries, Lseries], axis=1)

train_df, dummy_df = train_test_split(df, train_size=.95, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])     
print('train_df length:', len(train_df), 'test_df length:', len(test_df), 'valid_df length:', len(valid_df))

# gets the number of classes and the images count for each class in train_df
classes = sorted(list(train_df['labels'].unique()))
class_count = len(classes)
print('The number of classes in the dataset is:', class_count)

groups = train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))

countlist = []
classlist = []

for label in sorted(list(train_df['labels'].unique())):
    group = groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

# gets the classes with the minimum and maximum number of train images
max_value = np.max(countlist)
max_index = countlist.index(max_value)
max_class = classlist[max_index]

min_value = np.min(countlist)
min_index = countlist.index(min_value)
min_class = classlist[min_index]

print(max_class, 'has the most images =', max_value) 
print(min_class, 'has the least images =', min_value)

# gets the average height and width of the training images
ht = 0
wt = 0

train_df_sample = train_df.sample(n = 100, random_state = 123, axis=0) # select 100 random samples of train_df

for i in range (len(train_df_sample)):
    fpath = train_df_sample['filepaths'].iloc[i]
    img = plt.imread(fpath)
    shape = img.shape
    ht += shape[0]
    wt += shape[1]
    
print('average height= ', ht//100, ' average width= ', wt//100, 'aspect ratio= ', ht/wt)

# TRIMMING THE DATASET
# The maximum amount samples in a given class becomes 50

def trim(df, max_samples, min_samples, column):
    df = df.copy()
    groups = df.groupby(column)  # splits data by a certain criteria 
    trimmed_df = pd.DataFrame(columns = df.columns)
    groups = df.groupby(column)

    for label in df[column].unique(): 
        group = groups.get_group(label)
        count = len(group)    

        if count > max_samples:  # if a sign has less than x samples, add random samples until it is x
            sampled_group = group.sample(n = max_samples, random_state = 123, axis = 0)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis = 0)

        else: 
            if count >= min_samples:
                sampled_group = group        
                trimmed_df = pd.concat([trimmed_df, sampled_group], axis = 0)
    
    print('After trimming, the maximum samples in any class is now', max_samples)
    print('The minimum samples in any class is', min_samples)
    return trimmed_df

max_samples = 50 # increase or decrease depending on how long you want epochs to be vs accuracy 
min_samples = 0
column = 'labels'
train_df = trim(train_df, max_samples, min_samples, column)

# BALANCE THE DATASET

def balance(df, n, working_dir, img_size):

    def augment(df,n, working_dir, img_size):
        aug_dir = os.path.join(working_dir, 'aug')
        os.mkdir(aug_dir)        

        for label in df['labels'].unique():    
            dir_path = os.path.join(aug_dir,label)    
            os.mkdir(dir_path)

        # create and store the augmented images  
        total = 0
        gen = ImageDataGenerator(horizontal_flip = True, rotation_range = 20, width_shift_range = .2, 
                                 height_shift_range = .2, zoom_range = .2)
        groups = df.groupby('labels')  # group by class

        for label in df['labels'].unique():  # for every class               
            group = groups.get_group(label)  # a dataframe holding only rows with the specified label 
            sample_count = len(group)   # determine how many samples there are in this class  

            if sample_count < n: # if the class has less than target number of images
                aug_img_count = 0
                delta = n - sample_count  # number of augmented images to create
                target_dir = os.path.join(aug_dir, label)  # define where to write the images
                msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
                print(msg, '\r', end = '') # prints over on the same line
                aug_gen = gen.flow_from_dataframe(group, x_col = 'filepaths', y_col = None, target_size = img_size,
                                                class_mode = None, batch_size = 1, shuffle = False, 
                                                save_to_dir = target_dir, save_prefix = 'aug-', color_mode = 'rgb',
                                                save_format = 'jpg')
                
                while aug_img_count < delta:
                    images = next(aug_gen)            
                    aug_img_count += len(images)
                total += aug_img_count

        print('Total Augmented images created= ', total)

        # create aug_df and merge with train_df to create composite training set ndf
        aug_fpaths = []
        aug_labels = []
        classlist = os.listdir(aug_dir)

        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)     
            flist = os.listdir(classpath)   

            for f in flist:        
                fpath = os.path.join(classpath,f)         
                aug_fpaths.append(fpath)
                aug_labels.append(klass)

        Fseries = pd.Series(aug_fpaths, name = 'filepaths')
        Lseries = pd.Series(aug_labels, name = 'labels')
        aug_df = pd.concat([Fseries, Lseries], axis = 1)        
        df = pd.concat([df,aug_df], axis = 0).reset_index(drop = True)
        return df 
    
    df = df.copy() 

    # make directories to store augmented images
    aug_dir = os.path.join(working_dir, 'aug')   

    if 'aug' in os.listdir(working_dir):
        print('Augmented images already exist. To delete these and create new images enter D, else enter U to use these images', flush=True)
        ans = input(' ')

        if ans == 'D' or ans == 'd':            
            shutil.rmtree(aug_dir) # start with an clean empty directory  
            augment(df, n, working_dir, img_size)
            return df

        else:
            return df

    else:
        augment(df,n, working_dir, img_size)
        return df
        
   
n = 200 # number of samples in each class
working_dir = 'drive/MyDrive' # directory to store augmented images in, change to suit your file structure
img_size = (200,200) # size of augmented images
train_df = balance(train_df, n, working_dir, img_size)

# CREATE THE TRAIN_GEN, TEST_GEN, FINAL_TEST_GEN, AND VALID_GEN
batch_size = 30 # We will use and EfficientetB3 model, with image size of (200, 250) this size should not cause resource error
trgen = ImageDataGenerator(horizontal_flip = True, rotation_range = 20, width_shift_range = .2, height_shift_range = .2, zoom_range = .2)
t_and_v_gen = ImageDataGenerator() # 

msg = '{0:70s} for train generator'.format(' ')

print(msg, '\r', end = '') # prints over on the same line
train_gen = trgen.flow_from_dataframe(train_df, x_col = 'filepaths', y_col = 'labels', target_size=img_size,
                                   class_mode = 'categorical', color_mode = 'rgb', shuffle = True, batch_size = batch_size)

msg='{0:70s} for valid generator'.format(' ')

print(msg, '\r', end = '') # prints over on the same line
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col = 'filepaths', y_col = 'labels', target_size = img_size,
                                   class_mode = 'categorical', color_mode = 'rgb', shuffle = False, batch_size = batch_size)

# for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
# this insures that we go through all the sample in the test set exactly once.
length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]  

test_steps = int(length/test_batch_size)
msg='{0:70s} for test generator'.format(' ')
print(msg, '\r', end='') # prints over on the same line

test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col = 'filepaths', y_col = 'labels', target_size = img_size,
                                   class_mode = 'categorical', color_mode = 'rgb', shuffle = False, batch_size = test_batch_size)

# from the generator we can get information we will need later
classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())

class_count = len(classes)
labels = test_gen.labels

print('test batch size:', test_batch_size, 'test steps:', test_steps, 'number of classes:', class_count) 

# CREATE A FUNCTION TO SHOW SAMPLE TRAINING IMAGES

def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())    
    images, labels = next(gen) # get a sample batch from the generator 

    plt.figure(figsize=(20, 20))
    length = len(labels)

    if length < 25:   #show maximum of 25 images
        r = length
    
    else:
        r = 25

    for i in range(r):        
        plt.subplot(5, 5, i + 1)
        image = images[i]/255    
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')

    plt.show()
    
show_image_samples(train_gen)

# CREATE A MODEL USING TRANSER LEARNING WITH EfficientNetB3

img_shape = (img_size[0], img_size[1], 3)
model_name = 'EfficientNetB3'

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False, weights = "imagenet", input_shape = img_shape, pooling = 'max') 

# Note you are always told NOT to make the base model trainable initially- that is WRONG you get better results leaving it trainable
base_model.trainable = True

x = base_model.output
x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)

x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016), activity_regularizer=regularizers.l1(0.006),
                bias_regularizer = regularizers.l1(0.006), activation='relu')(x)

x = Dropout(rate = .4, seed = 123)(x) 
output = Dense(class_count, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = output)

lr = .001 # start with this learning rate
model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 

# CREATE A CUSTOM KERAS CALLBACK TO CONTINUE OR HALT TRAININGS

class ASK(keras.callbacks.Callback):

    def __init__ (self, model, epochs, ask_epoch): # initialization of the callback
        super(ASK, self).__init__()
        self.model = model               
        self.ask_epoch = ask_epoch
        self.epochs = epochs
        self.ask = True # if True query the user on a specified epoch
        

    def on_train_begin(self, logs = None): # this runs on the beginning of training
        if self.ask_epoch == 0: 
            print('you set ask_epoch = 0, ask_epoch will be set to 1', flush = True)
            self.ask_epoch = 1

        if self.ask_epoch >= self.epochs: # you are running for epochs but ask_epoch>epochs
            print('ask_epoch >= epochs, will train for', epochs, 'epochs', flush=True)
            self.ask = False # do not query the user

        if self.epochs == 1:
            self.ask = False # running only for 1 epoch so do not query user

        else:
            print('Training will proceed until epoch', ask_epoch,' then you will be asked to') 
            print(' enter H to halt training or enter an integer for how many more epochs to run then be asked again')  

        self.start_time= time.time() # set the time at which training started
        

    def on_train_end(self, logs = None):   # runs at the end of training     
        tr_duration = time.time() - self.start_time   # determine how long the training cycle lasted         
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg, flush = True) # print out training duration time
        

    def on_epoch_end(self, epoch, logs = None):  # method runs on the end of each epoch
        if self.ask: # are the conditions right to query the user?

            if epoch + 1 == self.ask_epoch: # is this epoch the one for quering the user?
                print('\n Enter H to end training or an integer for the number of additional epochs to run then ask again')
                ans = input()
                
                if ans == 'H' or ans == 'h' or ans == '0': # quit training for these conditions
                    print('you entered', ans, 'Training halted on epoch', epoch + 1, 'due to user input\n', flush = True)
                    self.model.stop_training = True # halt training

                else: # user wants to continue training
                    self.ask_epoch += int(ans)

                    if self.ask_epoch > self.epochs:
                        print('\nYou specified maximum epochs of as', self.epochs, 'cannot train for', self.ask_epoch, flush = True)
                    
                    else:
                        print ('you entered', ans, 'Training will continue to epoch', self.ask_epoch, flush = True)
                        
# INSTANTIATE CUSTOM CALLBACKS AND CREATE TWO TO CONTROL LEARNING RATE AND EARLY STOP

epochs = 40
ask_epoch = 5 # I found that 15 epochs strikes a good balance; it ends up having a training time of 4 hours
# 40 epochs is difficult to sustain, as inputs must be fed in throughout the day and the computer has to keep running for the full length of time
ask = ASK(model, epochs, ask_epoch)

rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 2, verbose = 1)
estop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 4, verbose = 1,restore_best_weights = True)

callbacks = [rlronp, estop, ask]

# TRAIN THE MODEL 

history = model.fit(x = train_gen, epochs = epochs, verbose = 1, callbacks = callbacks, validation_data = valid_gen,
               validation_steps = None, shuffle = False, initial_epoch = 0)
               
# DEFINTE A FUNCTION TO PLOT THE TRAINING DATA

def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data

    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']

    Epoch_count = len(tacc) + start_epoch
    Epochs = []

    for i in range (start_epoch, Epoch_count):
        Epochs.append(i + 1)

    index_loss = np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]

    plt.style.use('fivethirtyeight')

    sc_label = 'best epoch = ' + str(index_loss + 1 + start_epoch)
    vc_label='best epoch = ' + str(index_acc + 1 + start_epoch)

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,8))

    axes[0].plot(Epochs, tloss, 'r', label = 'Training loss')
    axes[0].plot(Epochs, vloss, 'g', label = 'Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s = 150, c = 'blue', label = sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs,tacc, 'r', label = 'Training Accuracy')
    axes[1].plot (Epochs, vacc, 'g', label = 'Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s = 150, c = 'blue', label = vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout    
    plt.show()
    
tr_plot(history, 0)

# MAKE PREDICTIONS ON THE TEST SET

def predictor(test_gen, test_steps):
    y_pred = []
    y_true = test_gen.labels
    classes = list(train_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds = model.predict(test_gen, steps = test_steps, verbose = 1) # predict on the test set
    tests = len(preds)

    for i, p in enumerate(preds):
            pred_index = np.argmax(p)         
            true_index = test_gen.labels[i]  # labels are integer values

            if pred_index != true_index: # a misclassification has occurred                                           
                errors=errors + 1

            y_pred.append(pred_index)

    acc = (1 - errors / tests) * 100
    print(f'there were {errors} in {tests} tests for an accuracy of {acc:6.2f}')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)

    if class_count <= 30:
        cm = confusion_matrix(ytrue, ypred)
        # plot the confusion matrix

        plt.figure(figsize = (16, 10))
        sns.heatmap(cm, annot = True, vmin = 0, fmt = 'g', cmap = 'Blues', cbar = False)       
        plt.xticks(np.arange(class_count) + .5, classes, rotation = 90)
        plt.yticks(np.arange(class_count) + .5, classes, rotation = 0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    clr = classification_report(y_true, y_pred, target_names = classes, digits = 4) # create classification report
    print("Classification Report:\n----------------------\n", clr)
    return errors, tests

errors, tests = predictor(test_gen, test_steps)

# SAVE THE MODEL
subject = 'asl' 
acc = str((1 - errors / tests) * 100)
index = acc.rfind('.')
acc = acc[:index + 3]
save_id = subject + '_' + str(acc) + '.h5'  # rename this to however you see fit 
model_save_loc = os.path.join(working_dir, save_id)
model.save(model_save_loc)
print ('model was saved as', model_save_loc) 
