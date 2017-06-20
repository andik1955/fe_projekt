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
		boundary coordinates of Shapefile extent
	
	based on:
		https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html	
	
	'''
	from osgeo import ogr
	import os
	
	driver = ogr.GetDriverByName("ESRI Shapefile")
	aoi = inDriver.Open(pathToAoi, 0)
	layer = aoi.GetLayer()
	extent = layer.GetExtent()
	
	return extent
	


############################
# parse Satellite imagery to find matches with Shapefile boundaries
############################




def findImagery(pathToScenes, shapefileBoundaries=None, outFolder=None):
	'''Function to retrieve Satellite imagery within AOI-boundaries and copy that to specified Folder
	
	
	Args:
		AOI Boundary coordinates
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
	#xMax, xMin, yMax, yMin = shapefileBoundaries
	
	
	
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
									print element
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
													i = 1
													if(ULX < xMin and ULX+NCOLS*Cellsize > xMax and ULY > yMax and ULY-NROWS*Cellsize < yMin):
														src = pathToScenes+scene+'/'+outerFolder+'/'+data+'/'+tile
														dst = outFolder + '/' + stile
														copytree(src, dst)
														break
													





############################
# extract AOI on satellite imagery
############################


def clipScenes(pathToScenes, shapefileBoundaries):
	'''Function to clip Satellite imagery to AOI-boundaries and copy/overwrite that to specified Folder
	
	
	Args:
		AOI Boundary coordinates
		path to a folder that contains sentinel 2 tiles that overlap shapefileBoundaries
		path where clipped scenes should be saved to
	
	Returns:
		Scene speficication of GDAL
	
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
						# SAGA-TOOLS start here
						# import as sgrd
						os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+band))
						
						# clip grids
						os.system('saga_cmd grid_tools 31 -GRIDS %s -CLIPPED %s -EXTENT 2 -SHAPES %s' % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)))
						
						if i = 1:
							driver = gdal.GetDriverByName('SAGA')
							raster_dataset = gdal.Open(raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly), gdal.GA_ReadOnly)
							geo_transform = raster_dataset.GetGeoTransform()
							proj = raster_dataset.GetProjectionRef()
							band = raster_dataset.GetRasterBand(1)
							band = band.ReadAsArray())
							rows, cols = bands_data.shape
							
							i = i +1
						
						# export as ...
						
	return ncols, nrows, geo_transform, projection
						
############################
# rasterize vector-training-data
############################
						
def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
	'''Rasterize all the vectors in the given directory into a single image.
	
	
	returns labeld groundtruth data
	
	'''
	
	
	labeled_pixels = np.zeros((rows, cols))
	for i, path in enumerate(file_paths):
		label = i+1
		ds = create_mask_from_vector(path, cols, rows, geo_transform, projection, target_value=label)
		band = ds.GetRasterBand(1)
		labeled_pixels += band.ReadAsArray()
		ds = None
	return labeled_pixels
						
						
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
			
	
	
	
	
	# hier muss gesplittet werden --> zufaellige Auswahl von validation und training pixels