import cv2
import matplotlib.pyplot as plt
from copy import copy

def cascade(path_to_cascade):
    face_cascade = cv2.CascadeClassifier(path_to_cascade)
    return face_cascade

def detect_faces(cascade, image, scaleFactor = 1.1, return_faces=False): 
    '''
    :scaleFactor: то, во сколько раз мы уменьшаем наше окно поиска лиц или чего-либо еще при каждой итерации
    '''
    
    '''на всякий случай поработаем с копией'''
    image_copy = image.copy()

    '''копию картинки переводим в серый, потому что detectMultiScale
       берет на фход только серые изображения'''
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    '''Детектим лица с помощью haar classifier''' 
    faces = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    print("Лиц обнаружено: " + format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)

    return (image_copy, faces) if return_faces else image_copy

def compute_face_areas_with_percents(array_of_faces, image):
    '''This function computes rectangle area and the percentage of it.
    It does not work without function detect_faces, which provides the array of faces (coordinates)'''
    
    img_h, img_w = image.shape[:2] #we do not need rgb size
    img_area = img_h * img_w
    areas = []
    percentages_of_area = []
    for face in array_of_faces:
        w, h = face[2], face[3]
        area = w * h
        percent = area / img_area * 100
        percentages_of_area.append(percent)
        areas.append(area)
    return areas, percentages_of_area
