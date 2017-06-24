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
	dataset = None  # Close the file




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
														copytree(src, dst)
														break


# old, working function
'''
############################
# extract AOI on satellite imagery
############################


def clipScenes(pathToScenes, aoiPath):
	Function to clip Satellite imagery to AOI-boundaries and clean up folder
	
	
	Args:
		AOI Boundary coordinates
		path to a folder that contains sentinel 2 tiles that overlap shapefileBoundaries
	
	Returns:
		Metadata for clipped 10m grid
	
	based on SAGA GIS -Tools and GDAl CMDs

	
		
	

	import os

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
						print 'metadata of clipped scene successfully written to variable'
						i += 1
				
				# clean *.mgrd files	
				os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.mgrd')
				
				# delete TMP.sgrds
				print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat', '\n'
				os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat'))

	return metaData
'''


# clipping with including resampling



############################
# extract AOI on satellite imagery including resample process
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
						print 'importing: ', pathToScenes+tile+'/'+imgFolder+'/'+band, '\n'
						print 'clipping with: ', aoiPath, '\n'

						# create reference grid for resampling of coarse bands
						if (k == 1):
							os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'system'+'.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+ fn[:-3] + 'B03' +'.jp2'))
							k += 1

						# SAGA-TOOLS start here
						# import as sgrd
						os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+band))
						
						# resample resBands
						# read grid system from band nr 3
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





'''
############################
# resample coarse bands to 10m resolution
############################


def resampleBands(pathToScenes):
	Function to resample bands that from coarse to fine resolution 
	
	
	Args:
		path to clipped scenes
	
	Returns:
		none

	based on GDAl CMDs

	
		

	
	import os

	res = {'B05': 20, 'B06': 20, 'B07': 20, 'B8A': 20, 'B11': 20, 'B12': 20, 'B01': 60, 'B09': 60, 'B10': 60}
	resBands = ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09', 'B10')
	
	for tile in os.listdir(pathToScenes):
		for imgFolder in os.listdir(pathToScenes+tile):
			if imgFolder == 'IMG_DATA':
				for band in os.listdir(pathToScenes+tile+'/'+imgFolder):
					fn, ext = os.path.splitext(band)
					if fn.endswith(resBands):
						print 'resampling process for %s started', pathToScenes+tile+'/'+imgFolder+'/'+band, '\n'
						# gdal cmd resampling
						os.system('gdalwarp [--help-general] [--formats]
    -s_srs srs_def EPSG:32632 -t_srs srs_def] [-to "NAME=VALUE"]* [-novshiftgrid]
    [-order n | -tps | -rpc | -geoloc] [-et err_threshold]
    [-refine_gcps tolerance [minimum_gcps]]
    [-te xmin ymin xmax ymax] [-te_srs srs_def]
    [-tr xres yres] [-tap] [-ts width height]
    [-ovr level|AUTO|AUTO-n|NONE] [-wo "NAME=VALUE"] [-ot Byte/Int16/...] [-wt Byte/Int16]
    -srcnodata "0" -dstnodata "0"
    [-srcalpha|-nosrcalpha] [-dstalpha]
    -r near [-wm memory_in_mb] [-multi] [-q]
    [-cutline datasource] [-cl layer] [-cwhere expression]
    [-csql statement] [-cblend dist_in_pixels] [-crop_to_cutline]
    [-of format] [-co "NAME=VALUE"]* [-overwrite]
    [-nomd] [-cvmd meta_conflict_value] [-setci] [-oo NAME=VALUE]*
    [-doo NAME=VALUE]*
    %s %s')
'''



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

def createTrainingValidation(gtNumpyArray):
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
	print 'created arrays'
	
	for i in range(0,dim[2]):
		print 'started loop'
		data = gtNumpyArray[:,:,i]
		print 'a'
		
		# tuple of indexes where array is nonzero
		isData = np.nonzero(data)
		print 'b'
		# extract first array from tuple, get length (== number of pixels that are nonzero)
		numberDataPixels = isData[0].shape[0]
		print 'c'
		#x = int(raw_input('%i pixels for class %s available. Please choose the number of training pixels.\nRemaining Pixels will be used for validation.'%(numberDataPixels, labelByIndex[i])))
		
		# choose random numbers from 0 to numberDataPixels
		# assign this part to training data
		train_idx = np.random.choice(numberDataPixels, size= 20000, replace = False)
		train_idx = train_idx.tolist()
		print 'd'
		# create validation data indices
		valid_idx=[]
		for i in range(0, numberDataPixels):
			if i in train_idx:
				continue
			else:
				valid_idx.append(i)
		print 'e'
		for i in train_idx:
			labeled_training[isData[0][i], isData[1][i]] += data[isData[0][i], isData[1][i]]
		print 'f'
		for i in valid_idx:
			labeled_validation[isData[0][i], isData[1][i]] += data[isData[0][i], isData[1][i]]
		train_idx = None
		
	return labeled_training, labeled_validation