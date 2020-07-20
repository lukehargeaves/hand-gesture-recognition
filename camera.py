import numpy as np
import os
from numpy import expand_dims
import cv2
# from keras_predict import pred
from trainer import predicter
from keras.preprocessing.image import img_to_array
import time
from im_to_txt import m
from random import random 
import shutil

# Main class that contains the functions needed to train, detect and remove the training images
# stored within the system.
class Yolo_Model:
    # Sets the intial parameters used within the definition of the class.  
    def __init__(self):
        # Enables the video capture
        self.cap = cv2.VideoCapture(0)
        # Sets the temporary directory used for the captured frame
        self.filename = "img.jpg"
        # Sets the destination of the training directory
        self.train_dir = "new_training"
        # Trys to make the training directory if it does not exist
        try:
            os.path.isfile(self.train_dir)
        except:
            os.mkdir(self.train_dir)
    # This function is called by model trainer. It is used to determine whether the detected
    # label is correct and if so store the image and convert the detected parameters into yolo format
    # to be used for further training the system.
    def func(self,curr):
        # Initialise a string used to store the boolean from the input.
        boo = ""
        # Read the stored frame captured.
        image  = cv2.imread(self.filename)
        # This removes the stored frame from the device so it no longer exists. This is a non-functional
        # additonal secruity measure.
        os.remove(self.filename)
        # Prompts the user to enter "T" if the predicted label is correct or "F" if it is incorrect.
        print("Enter 'T' if the label is correct or 'F' is the label is incorrect!")
        # Shows the user the predicted label
        boo = input(curr[0]+"\n")
        # Try and except incase an unexpected value was used within the input.
        try:
            if boo == 'T':
                # Selects a random number and converts it into a string and adds the .jpg file extension.
                # This method was used to reduce the chance of a duplicate filename when saving the training
                # images. The filename is then joined to the training directory.
                name = str(random())+ '.jpg'
                name = os.path.join(self.train_dir,name)
                # Try and except used incase the system could not write the image to the given location.
                try:
                    cv2.imwrite(name,image)
                except:
                    print("Could not save the image to " + str(name))
                # This calls the main main function within im_to_txt.py. This function converts the parameters
                # given into a yolo format and stores the data as a txt file, the same way thats needed for
                # training a yolo model. It passes the filename of the image so the txt filename can match it and
                # the array of parameters passed from the model detection, containing the bounding box locations.
                m(name, curr)

            # image  = cv2
        # Except block incase the value entered was not recognised.
        except ValueError as e:
            print(e)
            print("Value not recognised")

    # This function is used to detect the gesture presented.
    def model_detector(self):
        # While the camera is on
        while(self.cap.isOpened()):
            # Read the frame and frame number
            ret, frame = self.cap.read()
            if ret==True:
                # Sleep timer can be used inbetween predictions but is turned off for real time predictions.
                # time.sleep(1)
                # Flips the image so it is of the correct orientation.
                frame = cv2.flip(frame,+1)
                # Shows the captrued frame to the user. This image is updated for each frame.
                cv2.imshow('window',frame)
                # Try and except block used incase the image could not be written to the file location.
                try:
                    cv2.imwrite(self.filename,frame)
                except:
                    print("Could not save the image to img.jpg")
                # This calls the moodel predict function on the captured frame. It returns if a gesture could be 
                # located and the given parameters for that gesture. A higher thresh-hold of 80% certainty within the
                # prediction is used to ensure the prediction is accurate.
                output = predicter(0.6, self.filename)

                # This removes the locally saved frame from the device. 
                os.remove(self.filename)
                
                # If the length of the array returned from the model prediction is greater than 5,i.e. a prediction was made
                # then the first 6 values, corrosponding to the length of one prediction, will be upacked until there are no more
                # predictions from that frame. 
                while (len(output) >= 5): 
                    # The first 6 values corrosponding to the first prediction are extracted 
                    curr = output[0:6]
                    # The predicted label is the first value within the returned array
                    output = output[6:]
                    label = curr[0]
                    # A try block is used incase the given parameters for the bounding box are incorrect or cannot be labeled
                    # within the image. 
                    try:
                        # The annotations and labels are applied to the image. This produces a rectangele and label for the given
                        # detection on the captured frame.
                        image = cv2.rectangle(frame,(curr[1],curr[4]),(curr[3],curr[2]),(0,0,255),3)
                        image = cv2.putText(frame,label,(curr[1],(curr[2]-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    except:
                        print("Could not annote the image. Please ensure the co-ordinates are correct")    
                    cv2.imshow('window',image)

            # Used to detstroy the image capture window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break    
    # This function is used to train the system. It detects a gesture being shown by the user and applies
    # a label. It then prompts the user, asking if the label identified was correct. If yes, the yolo format
    # text file is created, needed for training, and the image is saved into the training directory. The user
    # is also shown the detected image with the label and bounding box. 
    def model_trainer(self):
        # Constant loop for capturing frames for the webcam. 
        while(True):
            # Captures the image and frame number.
            ret, frame = self.cap.read()
            if ret==True:
                # Additional sleep timer can be used for waiting in between capturing frames. Turned
                # off for real time detection. 
                # time.sleep(1)
                # Flips the frame so it appears the right way for the user.
                frame = cv2.flip(frame,+1)
                # Shows the user the frame that has been captured by the camera. This is constantly 
                # updated each time a new frame is captured.
                cv2.imshow('window',frame)
                # Try and except block used incase the system cannot write the captured frame. The frame
                # is saved so it can be used by the prediction file.
                try:
                    cv2.imwrite(self.filename,frame)
                except:
                    print("Could not save the image to img.jpg")
                # The predict method from trainer.py is called. This will return the prediction of the captured
                # image along with the the parameters needed for the boudnding box and prediction score. The parameter
                # 0.6 corrosponds to threshold of how confident the model is about the label applied. Any lower and the 
                # prediction will be discounted. 0.6 is lower than normal to allow for additonal classifcation to be made
                # during training.
                output = predicter(0.6,self.filename)
                # If the length of the array returned from the model prediction is greater than 5,i.e. a prediction was made
                # then the first 6 values, corrosponding to the length of one prediction, will be upacked until there are no more
                # predictions from that frame. 
                while (len(output) >= 5):
                    # The first 6 values corrosponding to the first prediction are extracted 
                    curr = output[0:6]
                    output = output[6:]
                    # The predicted label is the first value within the returned array
                    label = curr[0]
                    # A try block is used incase the given parameters for the bounding box are incorrect or cannot be labeled
                    # within the image. 
                    try:
                        image = cv2.rectangle(frame,(curr[1],curr[4]),(curr[3],curr[2]),(0,0,255),3)
                        image = cv2.putText(frame,label,(curr[1],(curr[2]-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    except:
                        print("Could not annote the image. Please ensure the co-ordinates are correct")    
                    cv2.imshow('window',image) 
                    # This tunction is called to check if the predicted label was correct and if so use the image for training.
                    self.func(curr)                                                                   
            # Used to destroy the image capture window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
# Adds additional ethics and security in removing the training directory
# from the system.
    def removeFiles(self):
        # Try and except used incase the training directory could not be located.
        try:
            # Removes the training directory.
            shutil.rmtree(self.train_dir, ignore_errors=True)
        # Throws an OS error as the file cannot be found, printing user instructions on the error.
        except OSError as e:
            print("Error: ")
            print(e)
# The main menu for the user to select the options for the system. This menu calls the functions
# used for training, detecting, and removing stored images.
    def main(self):
        menu = input("Press 1 to train the model: \nWarning training will store images of you!!\nPress 2 to detect the hand gesture: \nPress 3 to remove any images stored during training\n")
        if menu == '1':
            while(True):
                self.model_trainer()
        elif menu == '2':
            while(True):
                self.model_detector()
        elif menu == '3':
            self.removeFiles()
# The final else statement catches the program should a unexpected value be entered.
        else:
            print("Error. Correct option not selected")

# Automatically calls the main method within the Yolo_Model class
if __name__ == "__main__":
    model = Yolo_Model()
    # Call the main method 
    model.main()
    print("Closing application")
    # Try and except should the camera not have been used within the system.
    try:
        # Stop the frame capture
        cap.release()
        # Destroy the all the CV2 windows
        cv2.destroyAllWindows()
    except:
        print("Image capture not used")