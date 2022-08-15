# CREATE FOLDERS AND DOWNLOAD MEDIAPIPE
!mkdir captured_images
!mkdir annotated_images
!pip install mediapipe

# INSTALL PACKAGES
import os
import cv2
import h5py    
import keras 
import numpy as np    
import mediapipe as mp

from IPython.display import display, Javascript
from google.colab.output import eval_js
from IPython.display import Image
from base64 import b64decode

# SET UP NECESSARY VARIABLES
last_photo = sorted(os.listdir('captured_images'), reverse=True) # gets the last photo in the folder 
last_photo.remove('.ipynb_checkpoints') # This line will not always be required (see README for further details)

photo_num = 0

if last_photo == []: # if there aren't any images in the folder 
  print('No images currently in captured_images')
  
else: 
  last_photo = last_photo[0].replace('.jpg', '')
  photo_num = int(last_photo[-1]) + 1 # gets the number at the end of img name and adds 1

# CREATE A FUNCTION TO GET THE WEBCAM AND CAPTURE IMAGES
def take_photo(filename='captured_images/photo' + str(photo_num) + '.jpg', quality=0.8):

  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  
  with open(filename, 'wb') as f:
    f.write(binary)
  
  return filename

# CALL FUNCTION
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
  
# CONVERT CAPTURED IMAGES USING MEDIAPIPE
images = sorted(os.listdir('captured_images'))
images.remove('.ipynb_checkpoints')

names = []

for name in images: 
  name = name.replace('.jpg', '')
  names.append(name)

print(names)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

count = 0
IMAGE_FILES = 'captured_images/'
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

  for file in images: 
    file = IMAGE_FILES + images[count]

    print(file)

    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    
    image = cv2.flip(cv2.imread(file), 1)
    print(image) # ERROR OCCURING: IMAGES PRINT AS EMPTY?
    
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)

    if not results.multi_hand_landmarks:
      continue

    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
    cv2.imwrite(
        'annotated_images/annotated_image_' + names[count] + '.png', cv2.flip(annotated_image, 1))
    
    count += 1
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

      
# CALL MODEL (SEE OTHER PYTHON FILE FOR DETAILS)
reconstructed_model = keras.models.load_model("drive/MyDrive/asl2_89.72.h5")

# CREATE LIST OF CLASSES TO PREDICT WITH 
sdir = 'drive/MyDrive/processed_combine_asl_dataset' # training dataset, see README for further details
classlist = sorted(os.listdir(sdir))  # creates an alphabetical list of classes taken from sub-folder names 

# PREPARE IMAGES FOR PREDICTION
imgs = os.listdir('annotated_images')
imgs.remove('.ipynb_checkpoints') # This line will not always be required (see README for further details)

width = 400
height = 400

def image_reshape(images, width, height):
  reshaped_images = []
  dim = (width, height)

  for image in images: 
    img = cv2.imread('annotated_images/' + image)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    reshaped = resized.reshape(4, 200, 200, 3)
    reshaped_images.append(reshaped)

  return reshaped_images

reshaped_images = image_reshape(imgs, width, height)

# CREATE FUNCTION TO TRANSLATE PREDICTIONS INTO SIGN NAMES
def convert_prediction(array_pred, classlist):
  array_pred = array_pred.reshape(144)
  int_pred = np.argmax(array_pred)
  while int_pred >= 36: 
    int_pred -= 36

  pred = classlist[int_pred]

  return int_pred, pred

# PREDICT IMAGES
def predictions(reshaped_images, model, classlist):

  for image in reshaped_images:
    prediction = model.predict(image)
    converted_prediction = convert_prediction(prediction, classlist)
    print(converted_prediction)

predictions(reshaped_images, reconstructed_model, classlist)
