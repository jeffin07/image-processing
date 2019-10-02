import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def greenCircleDetect(image):
	green = [([65,60,60], [80,255,255])]
	for (lower, upper) in green:
	# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
 
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		return output

def redCircleDetect(image):
	red  = [([17, 15, 100], [50, 56, 200])]
	for (lower, upper) in red:
	# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
 
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		return output

def blueCircleDetect(image):
	blue = [([60, 31, 4], [220, 88, 50])]
	for (lower, upper) in blue:
	# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
 
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		return output




while True:
	ret, frame = cap.read()

	green_image = greenCircleDetect(frame)

	red_image = redCircleDetect(frame)

	blue_image = blueCircleDetect(frame)

	cv2.imshow("images", np.hstack([frame, red_image, green_image, blue_image]))
	# cv2.imshow('frame', frame)

	cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()
