# Imports
import cv2


import torch
from collections import defaultdict
from operator import itemgetter
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def register_photos(mtcnn, resnet, path_to_photos, database_directory, hand_label=False):
  '''
  Create a database of students by pointing to a folder with reference photos of all students (one photo per student).
  
  Args:
    path_to_photos (str): Path to the directory where reference photos are saved. There should be one photo per student.
    database_directory (str): Path to directory where the database will be saved in pickle format (.pkl)
    hand_label (bool): If True, function will display images one at a time and prompt user for label.
                       If False, function will label images based on filenames.

  Return:
    database (defaultdict): Dictionary of reference images and embeddings.
    Also saves database to database_directory specified in arguments.
  '''

  
  photo_names = glob.glob(os.path.join(path_to_photos, '*.jpg'))
  database = defaultdict(dict)

  # Label images
  if hand_label == True:
    for name in photo_names:
      
      # using mtcnn
      '''
      photo = Image.open(name)
      plt.imshow(photo)
      plt.axis('off')
      plt.show(block=False)
      '''
      # using cv2
      
      photo = cv2.imread(name)
      cv2.imshow(photo)
      id = input('Please enter student ID:')
       
      # Process images and store embeddings

      # using mtcnn
      '''
      photo = Image.open(name)
      face = mtcnn.forward(photo).to(device)  # replace with Ramya's face detection function
      '''
      # using cv2
      
      face = np.moveaxis(cv2.resize(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), (160, 160)), -1, 0)
      face = torch.Tensor((face - 127.5) / 128.0).to(device)
      
      vector = resnet.forward(face.unsqueeze(0)).cpu().detach().numpy()  # may need to be edited to match output from Ramya's function
        
      # Save vector in list to allow other reference vectors to be added on later (see update_database method in check_attendance below)
      # Save photo to allow for human comparison and attendance checking
      database[id]['vector'] = [vector]
      database[id]['photo'] = photo
    
    # Save database to file
    
    f = open(os.path.join(database_directory, 'database.pkl'), 'wb')
    pickle.dump(database, f)
    f.close()
    
    return database
  
  # Automatic labeling
  else:
    for name in photo_names:

      # using mtcnn
      '''
      photo = Image.open(name)
      face = mtcnn.forward(photo).to(device)   # replace with Ramya's face detection function
      '''
      # using cv2
      
      photo = cv2.imread(name)
      
      gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
      bboxes = face_cascade.detectMultiScale(gray, 
                                        scaleFactor=1.2, 
                                        minNeighbors=7, 
                                        minSize=(0, 0), 
                                        flags=cv2.CASCADE_SCALE_IMAGE)
      if len(bboxes) != 0:
         
          x, y, w, h = bboxes[0]
          photo = photo[y:y+h,x:x+w]
          
      face = np.moveaxis(cv2.resize(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), (160, 160)), -1, 0)
      face = torch.Tensor((face - 127.5) / 128.0).to(device)
      
      vector = resnet.forward(face.unsqueeze(0)).cpu().detach().numpy()  # may need to be edited to match output from Ramya's function
      database[name]['vector'] = [vector]
      database[name]['photo'] = photo
    
    f = open(os.path.join(database_directory, 'database.pkl'), 'wb')
    pickle.dump(database, f)    
    f.close()
    
    return database

def take_attendance(resnet, database, image, attendance):
  '''
  Processes one image at a time, comparing it to the database images to either find the closest match or decide no match is close
  enough and the person is unknown. Run this function on a for or while loop to process the images of all incoming students in a class.
  
  Args:
    database (defaultdict): Output from register_photos, it contains the images and embeddings of the reference images.
    face: uint8 numpy array. This is the image of the student face detected from the camera in class.

  Return:
    attendance (defaultdict): Dictionary of in-class photos, embeddings, and counts (for how many times the student was
                              counted present in that class period). The dictionary is updated each time the function is
                              called, and it is intended to be reset after each class period.
  '''

  # using mtcnn
  '''
  face = mtcnn.forward(image)   # replace with Ramya's face detection function
  '''
  # using cv2
  
  face = np.moveaxis(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (160, 160)), -1, 0)
  face = torch.Tensor((face - 127.5) / 128.0).to(device)
  
  vector = resnet.forward(face.unsqueeze(0)).cpu().detach().numpy()
  
  # Compare image embedding to reference embeddings. If multiple reference embeddings are available, take the average of the
  # cosine similarity to each of them.
  avg_similarity = []
  for student in database:
    similarity = []
    for ref_vector in database[student]['vector']:
      sim = np.dot(vector[0], ref_vector[0]) / (np.linalg.norm(vector) * np.linalg.norm(ref_vector))
      similarity.append(sim)
    average = sum(similarity) / len(similarity)
    avg_similarity.append((student, average))

  # Select most likely match. If it falls under 0.5 threshhold, identify student as "Unknown" with a unique digit suffix.
  # In either case, update attendance by incrementing count by one and saving unique embedding and original image.
  match = max(avg_similarity, key=itemgetter(1))
  if match[1] < .5:
    i=1
    while f'Unk_{i}' in attendance:
      i += 1
    match = f'Unk_{i}'
    attendance[match]['count'] = 1
    attendance[match]['vector_1'] = vector
    attendance[match]['photo_1'] = image
  else:
    match = match[0]
    attendance[match]['count'] += 1
    j = attendance[match]['count']
    attendance[match][f'vector_{j}'] = vector
    attendance[match][f'photo_{j}'] = image

  return attendance


if __name__ == "__main__":
    #import cv2
    from antispoofing_utils import FrameCounter, EmotionResNet, get_frame_data_dict, visualize_boxes
    import torchvision.transforms as T
    MIN_FACE_SIZE = (20, 20)
    MIN_EYE_SIZE = (0, 0)
    emo_size = (48, 48)
    model = EmotionResNet(learning_rate=None).load_from_checkpoint('emorec.ckpt', learning_rate=None).to(device)
    valid_tfms = T.Compose([T.ToPILImage(), T.Resize(emo_size), T.Grayscale(num_output_channels=3), T.ToTensor()])

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (fh, fw))
    frame_data_dict = get_frame_data_dict(model, valid_tfms, frame, MIN_FACE_SIZE, MIN_EYE_SIZE)
    x, y, x_max, y_max = frame_data_dict[0]['face_box']
    face = frame[x:x_max, y:y_max]
    face = np.moveaxis(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (160, 160)), -1, 0)
    face = torch.Tensor((face - 127.5) / 128.0).to(device)
    