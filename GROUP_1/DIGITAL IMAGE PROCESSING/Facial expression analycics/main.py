#--------------IMPORT REQUIRED LIBRARIES--------------------------
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

#----------------Pretrained model----------------------------------------------
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

#------------------EMOTIONAL LABLES------------------------------------
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#---------------------MAIN FUNCATION FOR DETECTION----------------------------
def start_detection():
    def detect_emotions():
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        panel.img = img
        panel.config(image=img)
        if detection_active:
            panel.after(10, detect_emotions)

    global detection_active
    detection_active = True
    #--------START DETECTION-------------------------
    detect_emotions()  

def stop_detection():
    global detection_active
    detection_active = False

#----------Tkinter setup--------------------
root = tk.Tk()
root.title("Real-time Emotion Detection")

#----------------OpenCV setup for video capture---------------------
cap = cv2.VideoCapture(0)

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

#--------------Button to start detection------------------------
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

#----------------------Button to stop detection------------------------
stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack()

root.mainloop()