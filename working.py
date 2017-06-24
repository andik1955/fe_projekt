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
			labeled_pixels = np.zeros((rows, cols, len(os.listdir(rasterPath))))
		
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
	
	
drive = 'E:/'
gtRasters = drive + 'VU_Fernerkundung/Projekt/input_rasterized/'

labeled_pixels, labelByIndex = loadRasters(gtRasters)