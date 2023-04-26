import torch
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import cv2
import mediapipe as mp

from ITrackerModel import ITrackerModel
from utils import *


MODEL_PATH = 'checkpoint.pth.tar'
LEFT_EYE_MEAN_PATH = 'mean_left_224.mat'
RIGHT_EYE_MEAN_PATH = 'mean_right_224.mat'
FACE_MEAN_PATH = 'mean_face_224.mat'
IMAGE_SIZE = (224, 224)
FACE_GRID_SIZE = (25, 25)
LEFT_EYE_MASK = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_MASK = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_IRIS_MASK = [474, 475, 476, 477]
RIGHT_IRIS_MASK = [469, 470, 471, 472]

ARROW_LENGTH = 200

# Load model from checkpoint
model_state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model = ITrackerModel()
model.load_state_dict(model_state['state_dict'])

# Load mean images
left_eye_mean = np.array(sio.loadmat(LEFT_EYE_MEAN_PATH)['image_mean'], dtype=np.float32)
right_eye_mean = np.array(sio.loadmat(RIGHT_EYE_MEAN_PATH)['image_mean'], dtype=np.float32)
face_mean = np.array(sio.loadmat(FACE_MEAN_PATH)['image_mean'], dtype=np.float32)

# Open webcam
cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Face', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Face mask', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Left eye', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Right eye', cv2.WINDOW_GUI_NORMAL)

cv2.resizeWindow('MediaPipe Face Mesh', 1280, 720)
cv2.resizeWindow('Face', IMAGE_SIZE[0], IMAGE_SIZE[1])
cv2.resizeWindow('Face mask', IMAGE_SIZE[0], IMAGE_SIZE[1])
cv2.resizeWindow('Left eye', IMAGE_SIZE[0], IMAGE_SIZE[1])
cv2.resizeWindow('Right eye', IMAGE_SIZE[0], IMAGE_SIZE[1])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)


def transform_image(image, mean):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        SubtractMean(mean),
    ])
    
    return transformation(image).unsqueeze(0)    
    
def get_prediction(face_image, left_eye_image, right_eye_image, face_grid):
    face_image = transform_image(face_image, face_mean)
    left_eye_image = transform_image(left_eye_image, left_eye_mean)
    right_eye_image = transform_image(right_eye_image, right_eye_mean)
    face_grid = torch.from_numpy(face_grid).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(face_image, left_eye_image, right_eye_image, face_grid)
     
    prediction = prediction.squeeze(0).numpy()
     
    return prediction
    

def get_bounding_box(keypoints_pixels, mask=None):
    if mask is None:
        mask = np.ones(keypoints_pixels.shape[0], dtype=bool)
    
    x_min, x_max = int(keypoints_pixels[mask,0].min()), int(keypoints_pixels[mask,0].max())
    y_min, y_max = int(keypoints_pixels[mask,1].min()), int(keypoints_pixels[mask,1].max())
    
    return x_min, x_max, y_min, y_max

def convert_to_square(x_min, x_max, y_min, y_max, axis='x', padding=None):
    if axis == 'x':
        x_center = (x_min + x_max) // 2
        x_min = x_center - (y_max - y_min) // 2
        x_max = x_center + (y_max - y_min) // 2

    elif axis == 'y':
        y_center = (y_min + y_max) // 2
        y_min = y_center - (x_max - x_min) // 2
        y_max = y_center + (x_max - x_min) // 2
    
    else:
        raise ValueError('axis must be either x or y')
    
    if padding is not None:
        width = x_max - x_min
        height = y_max - y_min
        
        x_min -= int(width * padding)
        x_max += int(width * padding)
        y_min -= int(height * padding)
        y_max += int(height * padding)

    return x_min, x_max, y_min, y_max

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        img_h, img_w = image.shape[:2]
        edited_image = image.copy()
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        face, left_eye, right_eye, face_grid = None, None, None, None
        
        prediction = np.array([0, 0])
        
        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                print('Multiple face can be seen!')
            
            face_landmarks = results.multi_face_landmarks[0]
            
            keypoints = np.array([[p.x, p.y, p.z] for p in face_landmarks.landmark])
            keypoints_pixels = np.multiply(keypoints, [img_w, img_h, 0])
            
            # Face
            x_min, x_max, y_min, y_max = convert_to_square(*get_bounding_box(keypoints_pixels), axis='x', padding=-0.1)
            face = image[y_min:y_max, x_min:x_max]
            
            edited_image = cv2.rectangle(edited_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Face mask
            face_grid = np.zeros((img_h, img_w), dtype=np.float32)
            face_grid[y_min:y_max, x_min:x_max] = 255

            # Left eye
            x_min, x_max, y_min, y_max = convert_to_square(*get_bounding_box(keypoints_pixels, LEFT_EYE_MASK), axis='y', padding=0.4)
            left_eye = image[y_min:y_max, x_min:x_max]
            
            edited_image = cv2.rectangle(edited_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Right eye
            x_min, x_max, y_min, y_max = convert_to_square(*get_bounding_box(keypoints_pixels, RIGHT_EYE_MASK), axis='y', padding=0.4)
            right_eye = image[y_min:y_max, x_min:x_max]
            
            edited_image = cv2.rectangle(edited_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            
            if face is not None and left_eye is not None and right_eye is not None and face_grid is not None:
                face_grid = cv2.resize(face_grid, FACE_GRID_SIZE)
                face = cv2.resize(face, IMAGE_SIZE)
                left_eye = cv2.resize(left_eye, IMAGE_SIZE)
                right_eye = cv2.resize(right_eye, IMAGE_SIZE)
                
                cv2.imshow('Face mask', cv2.flip(face_grid, 1))
                cv2.imshow('Face', cv2.flip(face, 1))
                cv2.imshow('Left eye', cv2.flip(left_eye, 1))
                cv2.imshow('Right eye', cv2.flip(right_eye, 1))
        
                prediction = get_prediction(face, left_eye, right_eye, face_grid)
            
                length_of_prediction = np.linalg.norm(prediction)
                normailized_prediction = prediction / length_of_prediction
                center_x, center_y = img_w // 2, img_h // 2
                
                print('Prediction: ', prediction)
                print('Normalized prediction: ', normailized_prediction)
                print('Length of prediction: ', length_of_prediction)
                
                edited_image = cv2.arrowedLine(edited_image, (center_x, center_y), (int(center_x + normailized_prediction[0] * ARROW_LENGTH), int(center_y + normailized_prediction[1] * ARROW_LENGTH)), (0, 0, 255), 2)
        
        # Display the image
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(edited_image, 1))
        
        # Exit the program
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
cap.release()
