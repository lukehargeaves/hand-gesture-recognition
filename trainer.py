import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# ****The following block of code was written by experiencor (2018) yolo3_one_file_to_detect_them_all [github]. Available
#  at https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py (Accessed: 19 April 2020).
#  with the following MIT License https://github.com/experiencor/keras-yolo3/blob/master/LICENSE allowing for the software
# to be used with no restrictions *****

# This code is used to decrypt the output from the Keras predict output, returning the label and bounding box locations for
# predictions that surpass the specified threshold. 

# Called to create a bounding box object for each detection. Used to store the x and y locations alongside the label and prediction
# strength. 	
class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes

	def get_axis(self):
		return xmin,xmax,ymin,ymax

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

# Decodes the Keras prediction. Seperates the array to return the paramets and class label
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):

	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	
	nb_class = (netout.shape[-1]) - 5
	boxes = []
	
	netout[..., :2]  = sigmoid(netout[..., :2])
	
	netout[..., 4:]  = sigmoid(netout[..., 4:])

	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[:,:,:, 5:] > obj_thresh
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

# Corrects the bounding boxes depending on the scale of the input image.
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

# Function for checking if the predction bounding boxes overlap
def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename):
	image = load_img(filename)
	original_width, original_height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=(416,416))
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image = image/ 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, original_width, original_height
# get all of the results above a threshold

# ***** THE REST OF THE CODE WAS WRITTEN BY ME *****


def get_info(boxes, labels, thresh):
	arr = []
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				# If so add the x and y locations and class labels along with the certainty of the prediction
				xmin = box.xmin
				ymin = box.ymin
				xmax = box.xmax
				ymax = box.ymax
				arr.append(labels[i])
				arr.append(xmin)
				arr.append(ymin)
				arr.append(xmax)
				arr.append(ymax)
				arr.append(box.classes[i]*100)
	return arr


 
expected_w, expected_h = 416,416
try:
    model = load_model('yolo.h5')
except FileNotFound as error:
    print(error)
    print('Model could not be loaded\n')
    print('Please ensure yolo.h5 is in the correct directory')
# photo_filename = 'img.jpg'
anchors =[[98,133,  80,210,  96,207, 130,188, 110,298, 172,294],[98,133,  80,210,  96,207, 130,188, 110,298, 172,294]]
labels = ["hand","thumbs_up","thumbs_down"]


# The main prediction class
def predicter(class_threshold,photo_filename):
	# Calls the load image pixel function to convert the input image into a pixel array. Returns the pixel array
	# for the image alongside the original width and height of input image.
	image, original_w, original_h = load_image_pixels(photo_filename)
	# Calls Keras' predict function. Returns an array of encoded predictions. 
	out = model.predict(image)
	# Creates an empty list called boxes. Used to store the box parameters from the predictions
	boxes = list()
	# Iterates through the output of the Keras predict array returned. Inside the returned array is the label, bounding
	# box locations, and score.
	for i in range(len(out)):
		# decode the output of the network
		boxes+=decode_netout(out[i][0], anchors[i], class_threshold, expected_h, expected_w)
	# correct the sizes of the bounding boxes for the shape of the image
	correct_yolo_boxes(boxes, original_h, original_w, expected_h, expected_w)
	# Function called to stop the system detecting overlapping bounding boxes within the prediction
	do_nms(boxes, 0.5)
	# get the details of the detected objects returing an array of the prediction label and bounding box locations/
	v_boxes = get_info(boxes, labels, class_threshold)
	return v_boxes