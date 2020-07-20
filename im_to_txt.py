# Converts the bouding box classification into the yolo format txt file
# to use for future training data.

# Main method that takes the filename for the image that is being converted
# alongside the array from the classification containing the data on the 
# prediction. 

# The main function.
def m(filename,arr):
	# Creates an empty array to store the correct YOLO parameters needed
	# within a YOLO text file for model training.
	txt=[]
	# Creates a text file using the same filename as the image. The .jpg
	# extension is removed from the .jpg file and replaced with .txt
	name = filename[0:-4]+'.txt'
	# Sets the height and width of the input image. The input images used
	# were 720 x 1280
	height = 720
	width = 1280
	# Calls the convert function. This turns the x and y bounding box locations
	# into the correct parameters needed for YOLO training
	x1,y1,x2,y2= converter(height,width,arr)
	# Converts the text label into a numerical label corrosponding to the expected
	# class.
	l = label_to_num(arr)

	# Adds the prediction label and the YOLO bounding box locations to the text array
	txt.append([name,l,x1,y1,x2,y2])
	# Writes the text array to the text file 
	for i in range(len(txt)):
	  f = open(name,"w")
	  f.write(' '.join(repr(e) for e in txt[i][1:]))
	  f.close()

# Function to convert string label to a numerical label
def label_to_num(arr):
  if arr[0] == 'hand':
    return 0
  elif arr[0] == 'thumbs_up':
    return 1
  elif arr[0] == 'thumbs_down':
    return 2

# Converts the x and y bounding box locations to YOLO format. 
def converter(height,width, arr):
	xmin = arr[1]
	ymin = arr[2]
	xmax = arr[3]
	ymax = arr[4]
	x = (float((xmin + xmax)) / 2) / width
	y = (float((ymin + ymax)) / 2) / height

	w = float((xmax - xmin)) / width
	h = float((ymax - ymin)) / height
	return x,y,w,h

