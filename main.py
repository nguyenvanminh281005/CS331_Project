from deepface import DeepFace
import cv2

result = DeepFace.verify(img1_path = "../dataset/biden1.jpg", img2_path = "../dataset/biden2.jpg")