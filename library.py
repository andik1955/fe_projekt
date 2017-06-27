'''
Collection of functions to preprocess and classify satellite imagery

Author: Andreas Kollert
'''


############################
# get AOI
############################

def getAOI(pathToAoi):
	'''Function to retrieve AOI boundaries
	
	
	Args:
		path to Shapefile with AOI
	
	Returns:
		boundary coordinates of Shapefile extent as list of four coordinates
		xMin, xMax, yMin, yMax
	
	based on:
		https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html	
	
	'''
	from osgeo import ogr
	
	driver = ogr.GetDriverByName("ESRI Shapefile")
	aoi = driver.Open(pathToAoi, 0)
	layer = aoi.GetLayer()
	extent = layer.GetExtent()
	
	extent = list(extent)
	return extent


############################
# get grid metadata
############################

def getMeta(pathToGrid):
	'''Function to retrieve metadata of a grid
	
	
	Args:
		path to grid
	
	Returns:
		see return...
	
	based on:
		https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html	
	
	'''
	from osgeo import gdal
	
	grid = gdal.Open(pathToGrid)
	
	ncols = grid.RasterXSize
	nrows = grid.RasterYSize
	geo_transform = grid.GetGeoTransform()
	projection = grid.GetProjection()
	
	
	return ncols, nrows, geo_transform, projection


############################
# write array to geotiff
############################

def write_geotiff(fname, outFolder, data, geo_transform, projection):
	'''Create a GeoTIFF file with the given data.	
	
	'''	
	
	from osgeo import gdal
	
	fn = fname + '.tif'
	
	driver = gdal.GetDriverByName('GTiff')
	rows, cols = data.shape
	dataset = driver.Create(outFolder + fn, cols, rows, 1, gdal.GDT_Byte)
	dataset.SetGeoTransform(geo_transform)
	dataset.SetProjection(projection)
	band = dataset.GetRasterBand(1)
	band.WriteArray(data)
	dataset = None

############################
# get class values/labels
############################	

def getClassValue(pathToClasses):
	''' Return class values/labels as dictionary
	
	'''
	
	import os
	labelByValue = {}
	for cl in os.listdir(pathToClasses):
		fn, ext = os.path.splitext(cl)
		labelByValue[fn[-1:]] = fn
	
	return labelByValue
	
############################
# parse imagery and copy tiles that contain aoi boundary
############################

def findImagery(pathToScenes, shapefileBoundaries=None, outFolder=None):
	'''Function to retrieve Satellite imagery containing AOI and copy that to specified Folder
	
	
	Args:
		path to a folder that contains unzipped sentinel 2 imagery as downloaded
		AOI Boundary as xMin, xMax, yMin, yMax tuple
		path where output should be copied to
	
	Returns:
		None
	'''
	
	import os
	import xml.etree.ElementTree as ET
	from shutil import copytree
	
	xMin, xMax, yMin, yMax = shapefileBoundaries
		
	if os.path.exists(outFolder):
		print 'outFolder %s already exits. Script terminates.'%(outFolder)
	else:
		for scene in os.listdir(pathToScenes):
			name, extension = os.path.splitext(scene)
			if (extension == '.zip'):
				continue
			for outerFolder in os.listdir(pathToScenes+scene):
				for data in os.listdir(pathToScenes+scene+'/'+outerFolder):
					if (data == 'GRANULE'):
						for tile in os.listdir(pathToScenes+scene+'/'+outerFolder+'/'+data):
							for element in os.listdir(pathToScenes+scene+'/'+outerFolder+'/'+data+'/'+tile):
								path = pathToScenes+scene+'/'+outerFolder+'/'+data+'/'+tile+'/'
								fn, ext = os.path.splitext(element)
								
								if (ext == '.xml'):
									tree = ET.parse(path+element)
									root = tree.getroot()
									for child in root:
										for Tile_GeoCoding in child.findall('Tile_Geocoding'):
											for size in Tile_GeoCoding.findall('Size'):
												Resolution = int(size.attrib['resolution'])
												if(Resolution == 10):
													Cellsize = Resolution
													for nrows in size.findall('NROWS'):
														NROWS =  int(nrows.text)
													for ncols in size.findall('NCOLS'):
														NCOLS = int(ncols.text)
													for geopos in Tile_GeoCoding.findall('Geoposition'):
														for ulx in geopos.findall('ULX'):
															ULX = float(ulx.text)
														for uly in geopos.findall('ULY'):
															ULY = float(uly.text)
													
													if(ULX < xMin and ULX+NCOLS*Cellsize > xMax and ULY > yMax and ULY-NROWS*Cellsize < yMin):
														src = pathToScenes+scene+'/'+outerFolder+'/'+data+'/'+tile
														dst = outFolder + '/' + tile
														print 'copy\n', src, '\nto\n', dst, '\n'
														copytree(src, dst)
														break





############################
# clip imagery to AOI and resample coarse resolution bands
############################


def clipScenes(pathToScenes, aoiPath):
	'''Function to clip Satellite imagery to AOI-boundaries and clean up folder
	
	
	Args:
		AOI Boundary coordinates
		path to a folder that contains sentinel 2 tiles that overlap shapefileBoundaries
	
	Returns:
		Metadata for clipped 10m grid
	
	based on SAGA GIS -Tools and GDAl CMDs	
		
	'''
	
	import os
	
	resBands = ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09', 'B10')

	i = 1
	for tile in os.listdir(pathToScenes):
		for imgFolder in os.listdir(pathToScenes+tile):
			if imgFolder == 'IMG_DATA':
				k = 1
				for band in os.listdir(pathToScenes+tile+'/'+imgFolder):
					fn, ext = os.path.splitext(band)
					
					if ext == '.jp2':
						
						# create reference grid for resampling of coarse bands
						if (k == 1):
							os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'system'+'.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+ fn[:-3] + 'B03' +'.jp2'))
							k += 1
							
						print 'importing: ', pathToScenes+tile+'/'+imgFolder+'/'+band, '\n'
						print 'clipping with: ', aoiPath, '\n'						

						# import as sgrd
						os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+band))
						
						# resample resBands
						# read grid system from band nr 3/system.sgrd
						if fn.endswith(resBands):
							#bandKey = fn[-3:]
							os.system('saga_cmd grid_tools 0 -INPUT %s -OUTPUT %s -SCALE_UP 5 -TARGET_DEFINITION 1 -TARGET_TEMPLATE %s' % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+'system'+'.sgrd'))
						
						# clip grids
						os.system('saga_cmd grid_tools 31 -GRIDS %s -CLIPPED %s -EXTENT 2 -SHAPES %s' % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.sgrd',  aoiPath))

						# export grids as geotiff
						os.system('saga_cmd io_gdal 2 -GRIDS %s -FILE %s' % (pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.tif'))


						# clean up data *.sdat grids
						print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.sdat', '\n'
						os.system('gdalmanage delete -f SAGA %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.sdat'))
						# clean *.mgrd files	
						os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.mgrd')

						# clean up data *.jp2 grids
						print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ band, '\n'
						os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ band))
					
					if (i == 1 and fn.endswith('B03')):
						metaData = getMeta(pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.tif')
						print 'metadata of clipped scene successfully written to variable'
						i += 1
				
				# clean *.mgrd files	
				os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.mgrd')
				os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ 'system.mgrd')
				
				# delete TMP.sgrds
				print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat', '\n'
				os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat'))
				
				# delete system.sgrds
				print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+'system'+'.sdat', '\n'
				os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+'system'+'.sdat'))
	
	return metaData



############################
# rasterize vector layer, return raster/ target ds
############################

def rasterizeVectorData(vector_data_path, rasterized_data_path, cols, rows, geo_transform, projection):
	'''Rasterize vector data in given directory
	
	Args
		path to directory containing each class as a single vector layer
		path to output folder of rasterized layer
		grid system/geo_transform/projection --> set to clipped s2 imagery (output of clipScenes)		
	
	takes a vector layer, rasterizes it and dumps it in the folder specified by rasterized_data_path
	if vector layer is <wald.shp> then raster will be <wald.sgrd> in the given directory	
	a different raster format could be specified by driver (-->also file extension in driver.Create)

	returns dictionary with values corresponding to a class
	
	'''
	from osgeo import gdal
	import os
	
	labelByValue = {}

	for category in os.listdir(vector_data_path):
		if category.endswith('.shp'):
			print 'Rasterizing ', category, '...'
			
			shape = vector_data_path + category
			
			data_source = gdal.OpenEx(shape, gdal.OF_VECTOR)
			layer = data_source.GetLayer(0)
			
			# set driver of desired raster output format
			driver = gdal.GetDriverByName('GTiff')
			
			fn, ext = os.path.splitext(category)
			target_ds = driver.Create('%s%s.tif'%(rasterized_data_path, fn), cols, rows, 1, gdal.GDT_UInt16)
			target_ds.SetGeoTransform(geo_transform)
			target_ds.SetProjection(projection)
			
			# set noData value to 0
			band = target_ds.GetRasterBand(1)
			band.SetNoDataValue(0)
			
			# burn_values must be a sequence!
			bnValue = int(fn[-1:])
			
			gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[bnValue])
			print 'Successfully rasterized ', fn +ext
			
			labelByValue[bnValue] = fn
	
	# labelByValue --> values in output raster corresponding to a certain class
	return labelByValue


############################
# load gt rasters into numpy stack
############################

def loadRasters(rasterPath):
	''' Load rasterized Ground truth data to numpy array
	
	Args
		path to directory with rasterized classes
	
	Assumes that each raster contains unique values for its class and that theses classes do not overlap
		
	'''
	
	import os
	from osgeo import gdal
	import numpy as np	

	labelByIndex ={}
	
	for i, element in enumerate(os.listdir(rasterPath)):
		# get raster metadata from first raster in list
		if i == 0:
			cols, rows, geo_transform, projection = getMeta(rasterPath + element)
			labeled_pixels = np.zeros((rows, cols, len(os.listdir(rasterPath))), dtype = int)
		
		fn, ext = os.path.splitext(element)
		
		grid = gdal.Open(rasterPath + element)
		print 'gdal opened %s'%(element)

		band = grid.GetRasterBand(1)
		# band to array
		band_array = band.ReadAsArray()
		
		# add band_array to numpy stack
		labeled_pixels[:,:,i] = band_array
		print 'added %s to numpy stack'%(fn)
		labelByIndex[i] = fn
		
		grid = None

	print 25*'-', '\n\nsuccessfully loaded groundtruth data\n', 25*'-'
	
	# labelByIndex --> nth array in third dimension corresponding to a certain class (!= array values corresponding to certain class)
	return labeled_pixels, labelByIndex


############################
# create/process training/validation-data
############################

def createTrainingValidation(gtNumpyArray, trainPixSize=100):
	'''Function that creates
		-a random subset of GroundTruth data for training
		-a random subset of GroundTruth data for validation
		-choose sampla size of training and validation data
	
	Args:
		ground truth data in form of a numpy array, each band in 3rd dimension corresponds to a class
		training_size: size of training data in pixels
	
	Returns:
		labeled_training : 2D array with values corresponding to a certain class (= labels)
		labeled_validation : 2D array with values corresponding to a certain class (= labels)
	
	based on https://www.machinalis.com/blog/python-for-geospatial-data-processing/
	
	'''
	
	import numpy as np
	
	dim = gtNumpyArray.shape
	training = np.zeros((dim[0], dim[1]), dtype = int)
	validation = np.zeros((dim[0], dim[1]), dtype = int)
	
	
	for i in range(0,dim[2]):
		data = gtNumpyArray[:,:,i]
		
		# tuple of indexes where array is nonzero
		isData = np.nonzero(data)
		# extract first array from tuple, get length (== number of pixels that are nonzero)
		numberDataPixels = isData[0].shape[0]
		numberArray = np.arange(numberDataPixels)
		
		# choose random numbers from 0 to numberDataPixels
		# assign this part to training data
		train_idx = np.random.choice(numberDataPixels, size= trainPixSize, replace = False)
		
		mask_array = np.ones(numberDataPixels, dtype = int)
		mask_array[train_idx] = 0
		
		isValid = np.nonzero(mask_array)
		
		valid_idx = numberArray[isValid]
		
		xx = (np.asarray(isData[0][train_idx]), np.asarray(isData[1][train_idx]))
		training[xx] += data[xx]
		
		yy = (np.asarray(isData[0][valid_idx]), np.asarray(isData[1][valid_idx]))
		validation[yy] += data[yy]
		
		train_idx, valid_idx = None, None
		
	return training, validation





############################
# load Data to classify
############################


def loadS2(pathToScenes, cols, rows):
	'''Function to load multitemporal S2 Data
	
	
	Args:
		path to a folder that contains sentinel 2 tiles that overlap shapefileBoundaries
	
	Returns:
		numpy array with every band
	
		
	'''
	
	import os
	import numpy as np
	from osgeo import gdal
	
	certainBand = ('02', '03', '04', '08', '11', '12', '10')	
	
	bandList = []

	for tile in os.listdir(pathToScenes):
		for imgFolder in os.listdir(pathToScenes+tile):
			if imgFolder == 'IMG_DATA':
				for band in os.listdir(pathToScenes+tile+'/'+imgFolder):
					fn, ext = os.path.splitext(band)
					if fn.endswith(certainBand):
						grid = gdal.Open(pathToScenes+tile+'/'+imgFolder+'/'+ fn + '.tif')
						band = grid.GetRasterBand(1)
						# band to array
						band_array = band.ReadAsArray()
						bandList.append(band_array)
						grid, band, band_array = None, None, None
	
	data = np.dstack(bandList)
	print 'Successfully loaded S2Data'
	return data


	
############################
# wrapper for LinearSVM classifier
############################

def wrapSVM(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, trainPixList, labelByValue):
	''' Wrapper for SVM classification

	test several sample sizes
	
	'''
	import os
	import numpy as np
	import time
	from datetime import datetime
	
	dimensions = S2Data.shape[2]
	
	n_samples = rows*cols
	flat_pixels = S2Data.reshape((n_samples, dimensions))
	
	c = 1.0
	
	dt = datetime.utcnow()
	
	month = '%02d'%(dt.month)
	day = '%02d'%(dt.day)
	hour = '%02d'%(dt.hour)
	minute = '%02d'%(dt.minute)

	newPath = projectFolder + str(dt.year) + month + day + '_' + hour + minute + 'UTC_' + 'sampleSizeTest' + '/'

	os.mkdir('%s'%(newPath))
	
	fobjTime = open("%sprocessingTime.txt"%(newPath), "w")
	fobjTime.write('nrTrainPix\ttimeToFit\ttimeToPredict\taccuracyScore\n')
	
	fobj = open("%slogfile.txt"%(newPath), "w")
	
		
	# Support Vector Machines
	from sklearn import metrics
	from sklearn import svm

	clf = svm.LinearSVC(C = c)
	
	for i in trainPixList:
		fobj.write("%s\nLogfile for LinearSVM Processing of %i Trainingpixels and C = %1.2f\n%s\n"%(50*'*', i, c, 50*'*'))
		labels_training, labels_validation = createTrainingValidation(labeled_pixels, i)
		
		# write training and validation pixels
		write_geotiff('training_areas_%ipixPerClass'%(i), newPath, labels_training, geo_transform, projection)
		write_geotiff('validation_areas_%ipixPerClassTrain'%(i), newPath, labels_validation, geo_transform, projection)
		
		idx_Train = np.nonzero(labels_training)
		training_samples = S2Data[idx_Train[0], idx_Train[1]]
		training_labels = labels_training[idx_Train]
		
		#######################################################
		startClf = time.clock()
		clf.fit(training_samples, training_labels)
		
		difClf = time.clock() - startClf
		difClfMin = int(difClf//60)
		print difClfMin
		difClfSec = int(difClf%60)
		print difClfSec
		fobj.write("%s\n Training of classifier took %i min %i sec \n%s\n"%(50*'*', difClfMin, difClfSec, 50*'*'))
		#######################################################
		#######################################################
		startPred = time.clock()
		
		result = clf.predict(flat_pixels)
		
		difPred = time.clock() - startPred
		difPredMin = int(difPred//60)
		print difPredMin
		difPredSec = int(difPred%60)
		print difPredSec
		
		fobj.write("%s\n Prediciting the entire image took %i min %i sec \n%s\n"%(50*'*', difPredMin, difPredSec, 50*'*'))
		#######################################################
		
		classification = result.reshape((rows, cols))
		# write classification data
		write_geotiff('LinearSVM_%i_trainPix'%(i), newPath, classification, geo_transform, projection)
		
		idx_Val = np.nonzero(labels_validation)
		predicted_val = classification[idx_Val]
		validation_labels = labels_validation[idx_Val]

		predicted_train = classification[idx_Train]
		
		fobj.write("\n%s\nConfusion matrix (validation data):\n\n%s\n" % (50*'-', metrics.confusion_matrix(validation_labels, predicted_val)))
		
		fobj.write("\n%s\nConfusion matrix (training data):\n\n%s\n" % (50*'-', metrics.confusion_matrix(training_labels, predicted_train)))
		
		
		target_names = ['Class %s' % s for s in labelByValue.values()]
		fobj.write("\n%s\nClassification report (validation data):\n%s" %  (50*'-', metrics.classification_report(validation_labels, predicted_val, target_names=target_names)))
		
		a = metrics.accuracy_score(validation_labels, predicted_val)
		fobj.write("\n%s\nClassification accuracy (validation data): %f" %  (50*'-', a))
		
		fobj.write("\n%s\nClassification accuracy (training data): %f" %  (50*'-',metrics.accuracy_score(training_labels, predicted_train)))
		
		fobjTime.write('%i\t%f\t%f\t%f\n'%(i, difClf, difPred, a))
		
	fobj.close()
	fobjTime.close()
	print 'finished Training Pixel Variation'	
	
	
############################
# test svm Parameters C
############################

def svmParam(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, trainPixNr, labelByValue, cList):
	''' Check C SVM classification on specified number of training pixels per class

	Args
		same as in wrapSVM except for training Pixel Number (set by trainPixNr)
		List with values for c

	-overwrites existing output directory

	
	'''
	
	import os
	import numpy as np
	import time
	from datetime import datetime
	from sklearn import metrics
	from sklearn import svm
	
	trainPixSize = trainPixNr
	
	dimensions = S2Data.shape[2]
	
	dt = datetime.utcnow()
	month = '%02d'%(dt.month)
	day = '%02d'%(dt.day)
	hour = '%02d'%(dt.hour)
	minute = '%02d'%(dt.minute)
	
	newPath = projectFolder + str(dt.year) + month + day + '_' + hour + minute + 'UTC_' + 'svmParameterTest' + '/'
	
	os.mkdir('%s'%(newPath))	
	
	fobjTime = open("%sprocessingTime.txt"%(newPath), "w")
	fobjTime.write('C\ttimeToFit\ttimeToPredict\taccuracyScore\n')
	
	fobj = open("%slogfile.txt"%(newPath), "w")	
	fobj.write("%s\nLogfile for LinearSVC Processing of %i Trainingpixels\n%s\n"%(50*'*', trainPixSize, 50*'*'))
	
	labels_training, labels_validation = createTrainingValidation(labeled_pixels, trainPixSize)
	
	# write training and validation pixels
	write_geotiff('training_areas', newPath, labels_training, geo_transform, projection)
	write_geotiff('validation_areas', newPath, labels_validation, geo_transform, projection)
	
	n_samples = rows*cols
	flat_pixels = S2Data.reshape((n_samples, dimensions))
	
	idx_Train = np.nonzero(labels_training)
	training_samples = S2Data[idx_Train[0], idx_Train[1]]
	training_labels = labels_training[idx_Train]
	
	for i in cList:
		clf = svm.LinearSVC(C = i)		
		fobj.write("%s\nRun with C = %f \n%s\n"%(50*'*', i, 50*'*'))
		
	
		##########################################################################################################################
		startClf = time.clock()
		
		clf.fit(training_samples, training_labels)
		
		difClf = time.clock() - startClf
		difClfMin = int(difClf//60)
		difClfSec = int(difClf%60)

		fobj.write("%s\n Training of classifier took %i min %i sec \n%s\n"%(50*'-', difClfMin, difClfSec, 50*'-'))
		##########################################################################################################################
		##########################################################################################################################
		startPred = time.clock()
		
		result = clf.predict(flat_pixels)
		
		difPred = time.clock() - startPred
		difPredMin = int(difPred//60)
		difPredSec = int(difPred%60)
		
		fobj.write("%s\n Prediciting the entire image took %i min %i sec \n%s\n"%(50*'-', difPredMin, difPredSec, 50*'-'))
		##########################################################################################################################
		
		classification = result.reshape((rows, cols))
		# write classification data
		write_geotiff('LinearSVM_C%1.2f'%(i), newPath, classification, geo_transform, projection)
		
		idx_Val = np.nonzero(labels_validation)
		predicted_val = classification[idx_Val]
		validation_labels = labels_validation[idx_Val]
	
		predicted_train = classification[idx_Train]
		
		fobj.write("\n%s\nConfusion matrix (validation data):\n\n%s\n" % (50*'-', metrics.confusion_matrix(validation_labels, predicted_val)))
		
		#fobj.write("\n%s\nConfusion matrix (training data):\n\n%s\n" % (50*'-', metrics.confusion_matrix(training_labels, predicted_train)))
		
		
		target_names = ['Class %s' % s for s in labelByValue.values()]
		fobj.write("\n%s\nClassification report (validation data):\n%s" %  (50*'-', metrics.classification_report(validation_labels, predicted_val, target_names=target_names)))
		
		a = metrics.accuracy_score(validation_labels, predicted_val)
		fobj.write("\n%s\nClassification accuracy (validation data): %f\n\n" %  (50*'-', a))
		
		#fobj.write("\n%s\nClassification accuracy (training data): %f" %  (50*'-',metrics.accuracy_score(training_labels, predicted_train)))

		fobjTime.write('%f\t%f\t%f\t%f\n'%(i, difClf, difPred, a))
			
	fobj.close()
	fobjTime.close()
	print 'finished Parameter check'