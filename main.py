import cv2
import os
#import numpy as np
import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1, MTCNN

import time

from antispoofing_utils import FrameCounter, EmotionResNet, get_frame_data_dict, visualize_boxes
from antispoofing_utils import font, fontScale, lineType

from sim_utils import register_photos, take_attendance

from collections import defaultdict

emo_size = (48, 48)
MIN_FACE_SIZE = (20, 20)
MIN_EYE_SIZE = (0, 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
valid_tfms = T.Compose([T.ToPILImage(), T.Resize(emo_size), T.Grayscale(num_output_channels=3), T.ToTensor()])
model = EmotionResNet(learning_rate=None).load_from_checkpoint('emorec.ckpt', learning_rate=None).to(device)
resnet = InceptionResnetV1(pretrained='casia-webface', device=device).eval()
mtcnn = MTCNN(selection_method='center_weighted_size', device='cpu').eval()

database = register_photos(mtcnn, resnet, 'VGG_mini/ref', 'VGG_mini', hand_label=False)

attendance = defaultdict(dict)
for student in database:
    attendance[student]['count'] = 0

cap = cv2.VideoCapture(0)#'http://10.0.0.4:8080/video')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

if cap.isOpened():
    fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fh, fw = fh // 3, fw // 3

c = 0
info_text = 'Press S to start'
is_start, stage_one, stage_two = False, False, False
max_time = 5
start_time = 0
fc = FrameCounter()
skip_frame = 10
attendant = ''
    

while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if ret == True:
        c += 1
        if c >= skip_frame:
            frame = cv2.resize(frame, (fh, fw))
            frame_data_dict = get_frame_data_dict(model, valid_tfms, frame, MIN_FACE_SIZE, MIN_EYE_SIZE)
            visualize_boxes(frame, frame_data_dict)
            
            if len(frame_data_dict) != 0:
                if is_start:
                    if stage_one:
                        info_text = 'Blink! ' + str(time.time() - start_time)[0:4] + f'/{max_time}'
                        if time.time() - start_time < max_time:
                            blinked = fc.analyze_blink(frame_data_dict[0])
                            if blinked:
                                stage_one = False
                                stage_two = True
                                start_time = time.time()
                        else:
                            info_text = 'Fail! Try Again. Press S to start'
                            stage_one = False
                        
                    if stage_two:
                        info_text = 'Smile! ' + str(time.time() - start_time)[0:4] + f'/{max_time}'
                        if time.time() - start_time < max_time:
                            smiled = fc.analyze_emo(frame_data_dict[0])
                            if smiled:
                                info_text = 'Success! This is '
                                x, y, x_max, y_max = frame_data_dict[0]['face_box']
                                face = frame[x:x_max, y:y_max]
                                attendance = take_attendance(resnet, database, face, attendance)
                                attendant = [k for k, v in attendance.items() if v['count'] > 0][0]
                                attendant = os.path.basename(os.path.normpath(attendant)).replace('.jpg', '')
                                info_text += str(attendant) + '. Press S to start'
                                stage_two = False
                        else:
                            info_text = 'Fail! Try Again. Press S to start'
                            stage_two = False
                        
                    
            cv2.putText(frame, info_text, (0, 20), font, fontScale, (128, 255, 128), lineType)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if cv2.waitKey(25) & 0xFF == ord('s'):
                is_start = True
                stage_one = True
                start_time = time.time()
                info_text = 'Detecting Face...'
                
                
            cv2.imshow('Frame',frame)
            c = 0
    
    else: 
        break

cap.release()
cv2.destroyAllWindows()