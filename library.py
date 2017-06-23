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
# get tile metadata
############################

def getMeta(pathToTile):
	'''Function to retrieve metadata of a grid
	
	
	Args:
		path to grid
	
	Returns:
		see return...
	
	based on:
		https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html	
	
	'''
	from osgeo import gdal
	
	tile = gdal.Open(pathToTile)
	
	ncols = tile.RasterXSize
	nrows = tile.RasterYSize
	geo_transform = tile.GetGeoTransform()
	projection = tile.GetProjection()
	
	
	return ncols, nrows, geo_transform, projection


############################
# parse Satellite imagery to find matches with Shapefile boundaries
############################




def findImagery(pathToScenes, shapefileBoundaries=None, outFolder=None):
	'''Function to retrieve Satellite imagery within AOI-boundaries and copy that to specified Folder
	
	
	Args:
		AOI Boundary as xMin, xMax, yMin, yMax tuple
		path to a folder that contains sentinel 2 imagery as unzipped, as downloaded
		path where output should be copied to
	
	Returns:
		None
	'''
	
	import os
	import xml.etree.ElementTree as ET
	from shutil import copytree
	
	# get single coordinates
	# depends on structure of boundaries
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
									#print element
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
													#print NCOLS, NROWS, ULX, ULY
													# i = 1
													if(ULX < xMin and ULX+NCOLS*Cellsize > xMax and ULY > yMax and ULY-NROWS*Cellsize < yMin):
														src = pathToScenes+scene+'/'+outerFolder+'/'+data+'/'+tile
														dst = outFolder + '/' + tile
														print src
														#copytree(src, dst)
														break




############################
# extract AOI on satellite imagery
############################


def clipScenes(pathToScenes, aoiPath):
	'''Function to clip Satellite imagery to AOI-boundaries and copy/overwrite that to specified Folder
	
	
	Args:
		AOI Boundary coordinates
		path to a folder that contains sentinel 2 tiles that overlap shapefileBoundaries
		path where clipped scenes should be saved to
	
	Returns:
		Metadata for clipped 10m grid
	
	based on SAGA GIS -Tools
		
	'''
	
	import os
	from osgeo import gdal
	
	i = 1
	for tile in os.listdir(pathToScenes):
		for imgFolder in os.listdir(pathToScenes+tile):
			if imgFolder == 'IMG_DATA':
				for band in os.listdir(pathToScenes+tile+'/'+imgFolder):
					fn, ext = os.path.splitext(band)
					
					if ext == '.jp2':
						print 'importing: ', pathToScenes+tile+'/'+imgFolder+'/'+band, '\n'
						print 'clipping with: ', aoiPath, '\n'



						# SAGA-TOOLS start here
						# import as sgrd
						os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+band))
						
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
						i += 1
				
				# clean *.mgrd files	
				os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.mgrd')
				
				# delete TMP.sgrds
				print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat', '\n'
				os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat'))

	return metaData
						
						
############################
# rasterize vector layer, return raster/ target ds
############################

def rasterizeVectorData(vector_data_path, rasterized_data_path, cols, rows, geo_transform, projection):
	'''Rasterize the given vector
	
	Args
		grid system/geo_transform/projection --> set to clipped s2 imagery (output of clipScenes)
		path to directory containing each class as a single vector layer
		path to output folder of rasterized layer
	
	takes a vector layer, rasterizes it and dumps it in the folder specified by rasterized_data_path
	if vector layer is <wald.shp> then raster will be <wald.sgrd> in the given directory	
	a different raster format could be specified by driver (-->also file extension in driver.Create)

	returns dictionary with values corresponding to a class
	
	'''
	from osgeo import gdal
	import os
	
	gtRasters = []
	labelByValue = {}
	i = 1
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

			
			gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[i])
			
			print 'Successfully rasterized ', fn +ext
			
			grid_path = rasterized_data_path + fn + '.tif'
			gtRasters.append(grid_path)
			labelByValue[i+1] = fn
			i += 1
	# labelByValue --> values in output raster corresponding to a certain class
	return gtRasters, labelByValue


############################
# load gt rasters into numpy stack
############################


def loadRasters(rasterList):
	''' Load rasterized Ground truth data to numpy array
	
	Args
		List with raster paths
		for SAGA grids
	
	Assumes that each raster contains unique values for its class and that theses classes do not overlap
	
	
	'''
	import os
	from osgeo import gdal
	import numpy as np
	
	# get raster metadata from first raster in list
	cols, rows, geo_transform, projection = getMeta(rasterList[0])
	
	labeled_pixels = np.zeros((rows, cols, len(rasterList)))
	labelByIndex ={}
	
	for i, path in enumerate(rasterList):
		
		raster = os.path.basename(path)
		fn, ext = os.path.splitext(raster)
		
		grid = gdal.Open(path)
		print 'gdal opened %s'%(raster)

		band = grid.GetRasterBand(1)
		# band to array
		band_array = band.ReadAsArray()
		
		# add band_array to numpy stack
		labeled_pixels[:,:,i] = band_array
		print 'added %s to numpy stack'%(raster)
		labelByIndex[i] = fn
		
		grid = None

	print 25*'-', '\n\nsuccessfully loaded groundtruth data\n', 25*'-'
	
	# labelByIndex --> nth array in third dimension corresponding to a certain class (!= array values corresponding to certain class)
	return labeled_pixels, labelByIndex
	

############################
# create/process training/validation-data
############################

def createTrainingValidation(gtNumpyArray, labelByIndex trainingSize = 0):
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
	labeled_training = np.zeros((dim[0], dim[1]))
	labeled_validation = np.zeros((dim[0], dim[1]))
	
	
	for i in gtNumpyArray.shape[2]:
		
		data = gtNumpyArray[:,:,i]
		
		# tuple of indexes where array is nonzero
		isData = np.nonzero(data)
		
		# extract first array from tuple, get length (== number of pixels that are nonzero)
		numberDataPixels = isData[0].shape[0]
		
		x = int(raw_input('%i pixels for class %s available. Please choose the number of training pixels.\nRemaining Pixels will be used for validation.'%(numberDataPixels, labelByIndex[i])))
		
		# choose random numbers from 0 to numberDataPixels
		# assign this part to training data
		train_idx = np.random.choice(numberDataPixels, size= x, replace = False)
		
		# create validation data indices
		valid_list=[]
		for i in range(0, len(numberDataPixels)):
			if i in train_idx:
				continue
			else:
				valid_list.append(i)
		valid_idx =  np.asarray(valid_list)
		
		labeled_training += data[isData[0][train_idx], isData[0][train_idx]]
		
		labeled_validation += data[isData[0][valid_idx], isData[0][valid_idx]]

	return labeled_training, labeled_validation