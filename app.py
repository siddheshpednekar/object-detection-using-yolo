# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:23:51 2022

@author: User4
"""

# -*- coding: utf-8 -*-

from PIL import Image
import streamlit as st
image = Image.open('oimage.jpg')
import cv2
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



def run():
    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])

    if img_file is not None:
        st.image(img_file,use_column_width=False, width=1000)
        save_image_path = 'C:\\Users\\Admin\\Documents\\Downloads\\abc'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        result = save_image_path

        yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        with open('coco.names', 'r') as f:
          classes = f.read().splitlines()



        img = cv2.imread(result)

        plt.imshow(img)

        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)



        i = blob[0].reshape(320, 320, 3)
        plt.imshow(i)

        yolo.setInput(blob)

        output_layer_names = yolo.getUnconnectedOutLayersNames()

        layer_output = yolo.forward(output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_output:
          for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence >0.7:
               # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN

        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        indexes = np.array(indexes)

        for i in indexes.flatten():
          x, y, w, h = boxes[i]
          label = str(classes[class_ids[i]])
          confi = str(round(confidences[i], 2))
          color = colors[i]
          cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
          cv2.putText(img, label + " " + confi, (x, y+20), font, 2, color, 2)

        fig, ax = plt.subplots(figsize=(20,15))
        plt.imshow(img)
        cv2.imwrite('oimage.jpg', img)
        print('saved')

        class_output = []


        st.image('oimage.jpg', width=1000)
        for i in range(len(class_ids)):
            class_output.append(classes[class_ids[i]])
        df = pd.DataFrame({'classes':class_output})

        st.write(df.value_counts().rename_axis('class').reset_index(name='count')
)

run()