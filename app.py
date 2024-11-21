import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout

def crop_number_plate(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image

def find_contours(dimensions, img) :
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread(r'C:\Users\MRITH\Downloads\ANN_CA_3\Number-Plate-Recognition\contour.jpg')
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)
            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')
            plt.title('Predict Segments')
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0
            img_res.append(char_copy)
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)
    return img_res

def segment_characters(image) :
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def fix_dimension(img): 
    new_img = np.zeros((28, 28, 3), dtype=np.uint8)
    for i in range(3):
        new_img[:,:,i] = img
    return new_img
  
def show_results(char_images, loaded_model):
    dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    output = []
    for img in char_images:
        img_ = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        predictions = loaded_model.predict(img)
        y_ = np.argmax(predictions)
        character = dic[y_]
        output.append(character)  
    plate_number = ''.join(output)
    return plate_number

def get_exif_data(image):
    exif_data = {}
    try:
        image = Image.open(image)
        info = image._getexif()
        if info is not None:
            for tag, value in info.items():
                tag_name = TAGS.get(tag, tag)
                exif_data[tag_name] = value
    except Exception as e:
        print(f"Error: {e}")
    return exif_data

def get_geolocation(exif_data):
    gps_info = exif_data.get("GPSInfo", None)
    if not gps_info:
        return None
    gps_data = {}
    for tag, value in gps_info.items():
        tag_name = GPSTAGS.get(tag, tag)
        gps_data[tag_name] = value
    try:
        latitude = gps_data["GPSLatitude"]
        latitude_ref = gps_data["GPSLatitudeRef"]
        longitude = gps_data["GPSLongitude"]
        longitude_ref = gps_data["GPSLongitudeRef"]
        lat = convert_to_degrees(latitude)
        if latitude_ref != "N":
            lat = -lat
        lon = convert_to_degrees(longitude)
        if longitude_ref != "E":
            lon = -lon
        return lat, lon
    except KeyError:
        return None

def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def meta_data(image_path):
    exif_data = get_exif_data(image_path)
    geolocation = get_geolocation(exif_data)
    if geolocation:
        print(f"Latitude: {geolocation[0]}, Longitude: {geolocation[1]}")
    else:
        print("No GPS data found.")
    return geolocation

def detect_and_extract_number_plate(image):
    model = YOLO('C:/Users/MRITH/Downloads/ANN_CA_3/Number-Plate-Recognition/runs/detect/train/weights/best.pt')
    results = model.predict(source=image, save=False, imgsz=320, conf=0.5)
    cropped_image = None    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = crop_number_plate(image, (xmin, ymin, xmax, ymax))
    if cropped_image is None:
        return None, None
    char=segment_characters(cropped_image)
    char_models = Sequential()
    char_models.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
    char_models.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
    char_models.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
    char_models.add(Conv2D(128, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
    char_models.add(MaxPooling2D(pool_size=(4, 4)))
    char_models.add(Dropout(0.4))
    char_models.add(Flatten())
    char_models.add(Dense(128, activation='relu'))
    char_models.add(Dense(36, activation='softmax'))
    char_models.load_weights('C:/Users/MRITH/Downloads/ANN_CA_3/Number-Plate-Recognition/chars.weights.h5')
    a=show_results(char, char_models)
    print("done")
    return cropped_image,a



st.title('Number Plate Detection and Extraction')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = np.array(Image.open(uploaded_image))

    number_plate_image, extracted_text = detect_and_extract_number_plate(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)
    if uploaded_image:
        m = meta_data(uploaded_image)
        if m is not None:
            st.write(f"Latitude: {m[0]}")
            st.write(f"Longitude: {m[1]}")
        else:
            st.write("No GPS data found.")

    
    if number_plate_image is not None:
        st.image(number_plate_image, caption='Detected Number Plate', use_column_width=True)
        st.write(f"Extracted Text: *{extracted_text}*")
    else:
        st.write("No number plate detected.")