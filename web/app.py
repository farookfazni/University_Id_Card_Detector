from PIL import Image
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.optimizers import RMSprop
# from tkinter import *
# import skimage.io

# C:\Users\FarookFazni\anaconda3\Scripts\activate
# streamlit run app.py

pytesseract.pytesseract.tesseract_cmd = 'F:\\Program Files\\Tesseract-OCR\\tesseract.exe'

st.write("""
# Student Id details extracter
""")


class stdid():
    def __init__(self, img1):
        # img = cv2.imread(img1)
        # img = cv2.resize(img, (1029, 644))
        # kernel = np.ones((5,8))
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # res = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, kernel)
        bfilter = cv2.bilateralFilter(gray, 9, 50, 50)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        ret, thresh1 = cv2.threshold(
            edged, 0, 255, cv2.THRESH_BINARY)
        rect_kernel = cv2.getStructuringElement(cv2. MORPH_RECT, (25, 25))

        # Appplying dilation on the threshold image
        # dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        # erosion = cv2.erode(dilation,rect_kernel,iterations = 1)
        closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, rect_kernel)

        keypoints = cv2.findContours(
            closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        imagecont = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
        location = None
        for contour in contours:
            epsilon = 0.03*cv2.arcLength(contour, False)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                location = approx
                break
        width, height = 300, 400
        pts1 = np.float32([location[0], location[3], location[1], location[2]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
        imgId = cv2.warpPerspective(
            img1, matrix1, (width, height), borderValue=(255, 255, 255))
        imOut = cv2.cvtColor(imgId, cv2.COLOR_BGR2RGB)

        ret, thresh1 = cv2.threshold(
            edged, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        # Appplying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Creating a copy of image
        im2 = img1.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # x, y, w, h = 540, 220, 950, 580

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(
                bfilter, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = bfilter[y:y + h, x:x + w]

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(
                cropped, config='--oem 2 --psm 3')

        # plt.imshow(imgId)
        st.write(text)
        st.image(imgId)
        st.image(imagecont)
        # st.image(dilation)

        imOut = cv2.cvtColor(imgId, cv2.COLOR_BGR2RGB)
        cv2.imwrite('savedImage.jpg', imOut)

        # st.image(imOut, use_column_width=True,clamp = True)

        with open('file.txt', mode='w') as f:
            f.write(text)

def predict(frame):
    # dir_path2 = 'F:/CST17020/python/ImageProcessingProject/myProject/web/tflite_q_qwre_model4.tflite'
    dir_path = 'F:/CST17020/python/ImageProcessingProject/myProject'
    md = tf.keras.models.load_model(dir_path+'//'+'id_card_detection2')
    # with open(dir_path2,'rb') as f:
    #     tflite_q_qwre_model = f.read()
    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path=dir_path2)
    # interpreter.allocate_tensors()

    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    x = image.img_to_array(frame)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    # input_data = images
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # val = output_data
    val = md.predict(images)
    return val

file = st.file_uploader('Upload an Student ID image',
                        type=['jpg', 'png', 'jpeg'])
dir_path = 'F:/CST17020/python/ImageProcessingProject/myProject'

if file is not None:
    my_img = Image.open(file)
    st.image(my_img)
    frame = np.asarray(my_img)
    frame = cv2.resize(frame, (1029, 644))
    if st.button("Predict"):
        val = predict(frame)
        if val == 0:
            st.write("Id Card Detected")
            st.image(my_img)
            stdid(frame)
        elif val == 1:
            st.write("Please Provide a valid ID")
            
    
