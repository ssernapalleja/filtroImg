import numpy as np
import cv2
import logging
import os.path
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.DEBUG)
p_height = 6
p_width = 7



# read image file
# file_name_2D_points = '/mrt/Nutzer/Student/aguirre/Shared/thermal2test/png/Color/Color_01.png'
file_name_2D_points = '/mrt/Nutzer/Student/duenas/Shared/png/thermal (Kopie 1)/thermal_'
#file_name_2D_points = '/mrt/Nutzer/Student/aguirre/Shared/thermal2test/png/thermal/test.png'
#file_name_2D_points = '/mrt/Nutzer/Student/aguirre/Shared/thermal2test/png/Infrared_1/Infrared_1_04.png' # triangles
totalImg =0
total_buenas = 0;
lista_encontradas = []
for im_num in \
([10, 15,18,19,20]):
#range(104):#
	_num = str(im_num)
	if (im_num<10):
		_num = '0'+str(im_num)

	hacer = True
	while not os.path.isfile(file_name_2D_points+str(im_num)+'.png'):	
		hacer = False
		break
	if(hacer):
		
		im = np.float32(cv2.imread(file_name_2D_points+str(im_num)+'.png', 0))
		# original: normalized read image
		totalImg +=1
		im = (255.0 * (im - im.min())
		      / (im.max() - im.min())).astype(np.uint8)
		    
		#B = (1.0/25.0)*np.ones((5,5)) # Specify kernel here
		#C = cv2.filter2D(im, -1, B) # Convolve
		ret = False
		features = None

		cv2.imshow('im1', im)
		#cv2.waitKey()

		#C = cv2.GaussianBlur(im,(11,11),0)
		L = 1
		step = 1.0 / L
		# initialize grid for the circle matrix
		grid_circle = np.zeros((L * 2 + 1, L * 2 + 1))
		# Assign values to the grid from 1.0 to 0.0 as a circle representation,
		# where the center gets 1.0 and the radius gets 0.0
		for k in range(L):
			for i in range(L - k, L + k + 1):
				for j in range(L - k, L + k + 1):
					r = ((i - L) ** 2 + (j - L) ** 2) ** 0.5
					if r <= k:
						grid_circle[i, j] = 1
				
		#kernel = np.ones((5, 5), np.uint8)
		kernel = grid_circle.astype(np.uint8)
		#C = cv2.dilate(im, kernel, iterations=1)

		#C = cv2.equalizeHist(im)
		for _ in range(1):
			C = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				    cv2.THRESH_BINARY,11,-10)

			C2 = cv2.max(C,im)

			C = cv2.dilate(C2, kernel, iterations=1)
			cv2.imshow('im', C)
			#cv2.waitKey()

			cv2.imshow('im2', C)
			#cv2.waitKey()
			im = C

		edges = cv2.Canny(im,100,200)
		cv2.imshow('canny', edges)
		edges = cv2.max(edges,im)
		cv2.imshow('canny_edges', edges)
		im = edges
		
		

		im = cv2.GaussianBlur(im,(9,9),0)

		cv2.imshow('im_postFiltro', im)



		
		 
		# Set up the detector with default parameters.
		detector = cv2.SimpleBlobDetector_create()
		 
		# Detect blobs.
		keypoints = detector.detect(edges)
		 
		# Draw detected blobs as red circles.
		# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imshow("Keypoints", im_with_keypoints)

		im2 = im * 1
		for cycle in range(2):
		    logging.debug('Cycle... %d', cycle + 1)
		    if cycle == 1:
			logging.debug('Inverting image')
			im2 = 255 - im2

		    features = np.array([], np.float32)
		    # Since the findCirclesGrid algorithm for symmetric
		    # grid usually fails for a wrong height - width
		    # configuration, we invert here those parameters.
		    noEncontrado = False
		    for inner_cycle in range(2): 
			if inner_cycle == 0:
			    logging.debug('height - width')
			    ret, features = cv2.findCirclesGrid(im2, (p_height, p_width), features,
				                                cv2.CALIB_CB_SYMMETRIC_GRID)
			    if features is not None:
				cv2.drawChessboardCorners(im2, (p_height, p_width), features, True)
				cv2.imshow('im', im2)
				#cv2.waitKey()
				logging.debug('image:%d inner_cycle: %d', cycle + 1, inner_cycle + 1)
			    else:
				print('nothing')
			    if ret:
				print('siuu')
				noEncontrado = True
				total_buenas +=1
				lista_encontradas.append(im_num)
				break
			else:
			    logging.debug('width - height')
			    ret, features = cv2.findCirclesGrid(im2, (p_width, p_height), features,
				                                cv2.CALIB_CB_SYMMETRIC_GRID)
			    if features is not None:
				cv2.drawChessboardCorners(im2, (p_height, p_width), features, True)
				cv2.imshow('im3', im2)
				#cv2.waitKey()
				logging.debug('image:%d inner_cycle: %d', cycle + 1, inner_cycle + 1)
			    else:
				print('nothing')
			    if ret:
				print('siuu')
				noEncontrado = True
				total_buenas +=1
				lista_encontradas.append(im_num)
				# trasform the detected features
				# configuration to match the original
				# (height, width)
				features = features.reshape(p_height, p_width, 1, 2)
				features = np.transpose(features, (1, 0, 2, 3))
				features = features.reshape(p_width * p_height, 1, 2)
				break
		    if ret:
			break
		if(not noEncontrado):
			
			print(im_num)
		cv2.waitKey()
	    
print(str(total_buenas) + " de " + str(totalImg) + "  --  "+ str(100.0*total_buenas/totalImg))
print (lista_encontradas)
cv2.waitKey()
