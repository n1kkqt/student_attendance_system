import cv2
import numpy as np

import pytorch_lightning as pl

import torch
import torchmetrics
import torchvision.models as models


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
lineType = 2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FrameCounter:
    def __init__(self, blink_tol = 3, emo_tol = 3):
        self.blink_tol = blink_tol
        self.emo_tol = emo_tol
        
        self.n_frames_no_blink = 0
        self.n_frames_happy = 0
        self.n_frames_eyes_dissappeared = 0
        
    def analyze_blink(self, frame_data_dict):
            
        if len(frame_data_dict['face_box']) != 0 and len(frame_data_dict['eyes']) >= 1:
            self.n_frames_no_blink += 1
        else:
            self.n_frames_eyes_dissappeared += 1
        
        if self.n_frames_eyes_dissappeared > self.blink_tol:
            self.n_frames_no_blink = 0
            self.n_frames_eyes_dissappeared = 0
            return True
            
        return False
            
                
    def analyze_emo(self, frame_data_dict):
        
        if frame_data_dict['emotion'] == 'happy':
            self.n_frames_happy += 1
        else:
            self.n_frames_happy = 0
            
        if self.n_frames_happy > self.emo_tol:
            self.n_frames_happy = 0
            return True
            
        return False
            

class EmotionResNet(pl.LightningModule):
    def __init__(self, learning_rate):
      """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
      """
      super(EmotionResNet, self).__init__()
      
      resnet = models.resnet18(pretrained=True)
      resnet.fc = torch.nn.Linear(resnet.fc.in_features, 1)
      
      self.resnet = resnet
      self.activation = torch.nn.Sigmoid()
      self.accuracy_t = torchmetrics.Accuracy()
      self.accuracy_v = torchmetrics.Accuracy()
      self.learning_rate = learning_rate

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
      return optimizer

    def forward(self, x):
      x = self.activation(self.resnet(x))
      return x

    def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      y = y.to(torch.float)
      y_hat = self.activation(self.resnet(x)).squeeze(1)
      loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
      preds = (y_hat > 0.5).to(torch.int)
      self.log('train_loss', loss)
      self.log('train_acc_step', self.accuracy_t(preds, y.to(torch.int)))
      return loss

    def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      y = y.to(torch.float)
      y_hat = self.activation(self.resnet(x)).squeeze(1)
      loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
      preds = (y_hat > 0.5).to(torch.int)
      self.log('val_loss', loss)
      self.log('valid_acc_step', self.accuracy_v(preds, y.to(torch.int)))
      return loss

    def validation_epoch_end(self, validation_step_outputs):
      self.log('valid_acc_epoch', self.accuracy_v.compute())
      print('valid_acc_epoch', self.accuracy_v.compute())

    def training_epoch_end(self, outs):
      self.log('train_acc_epoch', self.accuracy_t.compute())
      print('train_acc_epoch', self.accuracy_t.compute())

def get_frame_data_dict(model, valid_tfms, frame, min_face_size, min_eye_size, margin=0):
    frame_data_dict = {}
    face_counter = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 
                                        scaleFactor=1.2, 
                                        minNeighbors=7, 
                                        minSize=min_face_size, 
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h,x:x+w]
        emotion, score = predict_emotion(model, valid_tfms, face_gray)
        y_init, h_init = y, h
        y = 0 if y - margin < 0 else y - margin
        h = h + margin if h + margin < frame.shape[0] else h
        frame_data_dict[face_counter] = {'face_box':[x, y, x+w, y+h], 'eyes':[], 'emotion':emotion, 'score':score}
        eyes = eye_cascade.detectMultiScale(face_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=7,
                                            minSize=min_face_size,
                                            flags = cv2.CASCADE_SCALE_IMAGE)
        eyes = np.array(eyes)#.reshape(-2, 4)
        if np.size(eyes) == 0:
            eyes = detect_eyes_separately(face_gray, min_eye_size)
        if np.size(eyes) != 0:
            for (ex, ey, ew, eh) in eyes:
                y, h = y_init, h_init
                frame_data_dict[face_counter]['eyes'].append([x+ex, y+ey, x+ex+ew, y+ey+eh])
        #cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
          
        face_counter += 1
    return frame_data_dict

def detect_eyes_separately(face_gray, min_eye_size):
    eyes = np.zeros((0,4))
    w, h = face_gray.shape
    face_left = face_gray[:, 0:w//2]
    face_right = face_gray[:, w//2:]
    left_eye = left_eye_cascade.detectMultiScale(
                    face_left,
                    scaleFactor=1.1,
                    minNeighbors=9,
                    minSize=min_eye_size,
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
                
    right_eye = right_eye_cascade.detectMultiScale(
                    face_right,
                    scaleFactor=1.1,
                    minNeighbors=9,
                    minSize=min_eye_size,
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
    
    left_eye = np.array(left_eye).reshape(-2, 4)
    right_eye = np.array(right_eye).reshape(-2, 4)
    eyes = np.append(eyes, left_eye, 0)
    eyes = np.append(eyes, right_eye, 0)
    eyes = np.atleast_2d(eyes).astype(int)
    return eyes

def visualize_boxes(frame, frame_data_dict):
    for face_data_dict in frame_data_dict.values():
        x, y, x_max, y_max = face_data_dict['face_box']
        emotion, score = face_data_dict['emotion'], face_data_dict['score']
        face_box_color = (255, 0, 0) if emotion == 'neutral' else (0, 255, 0)
        txt = emotion + ' ' + str(score)[0:4]
        cv2.rectangle(frame, (x,y),(x_max,y_max),face_box_color,2)
        cv2.putText(frame, txt, (x, y), 
                    font, fontScale, fontColor, lineType)
        for (x, y, x_max, y_max) in face_data_dict['eyes']:
            cv2.rectangle(frame, (x,y),(x_max,y_max),(0,0,255),2)


def predict_emotion(model, valid_tfms, img_array):
    img_array = valid_tfms(img_array).to(device)
    with torch.no_grad():
      score = model(img_array.unsqueeze(0)).cpu().numpy()[0][0]
    
    emotion = 'neutral' if score > 0.5 else 'happy'

    return emotion, score
