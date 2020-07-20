from trainer import predicter
import numpy as np
import os
import time 

# Function to evaluate the accuracy of the model.
def evaluate():

	# locates the directory of the testing images
	directory = "new_training"
	correct = 0
	total = 0
	# Tests to make sure the testing directory can be located. 
	try:
	    os.path.isfile(directory)
	except FileNotFoundError as e:
	    print("Testing directory does not exist")
	# Iterates through all of the files within the testing directory
	for filename in os.listdir(directory):
		if filename.endswith(".jpg"):
			name = os.path.join(directory,filename)
			# Calls the prediction on the image. Uses a threshold of 0.2 as this produced the max.
			# accuracy.
			output = predicter(0.4, name)
			# If a prediction is returned get the label from the returned prediction array.
			if len(output)>1:
				label =output[0]
			# Else use NA as the label was not detected
			else:
				label = "NA"
			# Runs the convert function, turning the string label to a numerical label. 
			label = convert(label)
			# Get the name of the YOLO file corrosponding to the testing image. This will contain
			# the label for the given test image.
			txt_name = filename[0:-3]
			txt_name = txt_name+'txt'
			txt_name = os.path.join(directory,txt_name)
			# Open the YOLO file in read mode.
			f = open(txt_name,'r')
			content = f.read()
			# Stotes the label of the image in the corr variable
			corr = content[0]
			# Incremenet the total number of predictions
			total += 1
			# If the predicted label matches the actual label, increment the correct total
			if corr == label:
				correct = correct +1
	# Once all the test images have tested, return the accuracy of the model. 
	print((correct/total)*100)

# Function to convert the string of the prediction to a numerical value.
def convert(label):
	if label == 'hand':
		return '0'
	elif label == 'thumbs_up':
		return '1'
	elif label == 'thumbs_down':
		return '2'

# Calls the main function
evaluate()