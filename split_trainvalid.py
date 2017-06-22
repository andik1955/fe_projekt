############################
# create/process training/validation-data
############################

def createTrainingValidation(shapePath, ratioTrainValid, rasterStack, rows, cols, geo_transform, projection):
	'''Function that creates
		-a random subset of GroundTruth data for training
		-a random subset of GroundTruth data for validation
		-choose ratio of training to validation data
	
	Args:
		path to ground truth shapeFolder
		ratio of training to validation data
		
		raster Stack with classification data
		raster metadata
	
	Returns:
		?
	
	based on https://www.machinalis.com/blog/python-for-geospatial-data-processing/
		
	'''
	
	import os
	import numpy as np
	import numpy.ma as ma
	
	files = [f for f in os.listdir(shapePath) if f.endswith('.shp')]
	classes = [f.split('.')[0] for f in files]
	
	shapefiles = [os.path.join(shapePath, f) for f in files if f.endswith('.shp')]
	
	# labeled pixel both training and validation data !
	labeled_ pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, projection)
	
	is_data = np.nonzero(labeled_pixels)
	data_labels = labeled_pixels[is_data]
	data_samples = rasterStack[is_data]
	
	rank = isdata[0].shape[0]
	
	train_idx = np.random.choice(rank, size= int(rank/2), replace = false)
	valid_list=[]
	for i in range(0, len(rank)):
		if i in train_idx:
			continue
		else:
			valid_list.append(i)
		
	valid_idx =  np.asarray(valid_list)
	
	for idx in train_idx:
		training_array = data_samples[isdata[0][idx], isdata[1][idx]]
	
	for idx in valid_idx:
		validation_array = data_samples[isdata[0][idx], isdata[1][idx]]
			
	
