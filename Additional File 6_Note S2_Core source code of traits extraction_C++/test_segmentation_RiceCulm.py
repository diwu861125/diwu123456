import numpy as np
import os.path
import scipy
import argparse
import scipy.io as sio
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import sys
import cv2 as cv
from skimage import data,filters,segmentation,measure,morphology,color


caffe_root = 'SegNet_RiceCulm/caffe-segnet/'				# Change this to the absolute directoy to caffe-segnet

sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
				args.weights,
				caffe.TEST)

input_shape = net.blobs['data'].data.shape

background = [0, 0, 0]
riceCulm = [255, 255, 255]

label_colours = np.array([background, riceCulm])

with open(args.data) as f:
	for line in f:
		line = line[:-1]
		line=line.strip().split(" ")
		origin=cv.imread(line[0])
		sub_x=0
		sub_y=0
		if origin.shape[1]%480==0:
			sub_x=int(origin.shape[1]/480)
		else:
			sub_x=int(origin.shape[1]/480+1)
		
		if origin.shape[0]%360==0:
			sub_y=int(origin.shape[0]/360)
		else:
			sub_y=int(origin.shape[0]/360+1)

		expand_width=sub_x*480
		expand_height=sub_y*360
		expand=np.zeros((expand_height,expand_width),dtype=np.uint8) 
		expand=cv.cvtColor(expand,cv.COLOR_GRAY2BGR)
		expand[0:origin.shape[0],0:origin.shape[1]]=origin
		expand_pred = np.zeros((expand_height,expand_width),dtype=np.uint8)
		expand_pred=cv.cvtColor(expand_pred,cv.COLOR_GRAY2BGR)

		for y in range(sub_y):
			for x in range(sub_x):
				input_image=expand[(y*360):(y*360+360),(x*480):(x*480+480)]
					
				input_image = input_image.transpose((2, 0, 1))
				input_image = input_image[(2, 1, 0), :, :]
				
				input_image = np.asarray([input_image])
				input_image = np.repeat(input_image, input_shape[0], axis=0)

				out = net.forward_all(data=input_image)

				predicted = net.blobs['prob'].data

				output = np.mean(predicted, axis=0)
				ind = np.argmax(output, axis=0)

				r = ind.copy()
				g = ind.copy()
				b = ind.copy()
				for l in range(0, 2):
					r[ind==l] = label_colours[l,0]
					g[ind==l] = label_colours[l,1]
					b[ind==l] = label_colours[l,2]

					segmentation_rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
					segmentation_rgb[:,:,0] = r
					segmentation_rgb[:,:,1] = g
					segmentation_rgb[:,:,2] = b
					
				expand_pred[(y*360):(y*360+360),(x*480):(x*480+480)]=segmentation_rgb
		
		pred=expand_pred[0:origin.shape[0],0:origin.shape[1]]
		
		kernel = np.ones((5,5),np.uint8)
		erosion = cv.erode(pred,kernel,iterations = 1)

		erosion = cv.cvtColor(erosion,cv.COLOR_RGB2GRAY)
		thresh=filters.threshold_otsu(erosion)
		bw=morphology.closing(erosion>thresh,morphology.square(3))
		dst=morphology.remove_small_objects(bw,min_size=500,connectivity=1)

		scipy.misc.toimage(dst, cmin=0.0, cmax=1.0).save(line[1])
		print ('Processed: ', line[0])

print ('Success!')
