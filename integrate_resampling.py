# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:14:08 2017

@author: csaq7453
"""

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
				for band in os.listdir(pathToScenes+tile+'/'+imgFolder):
					fn, ext = os.path.splitext(band)
					
					if ext == '.jp2':
						print 'importing: ', pathToScenes+tile+'/'+imgFolder+'/'+band, '\n'
						print 'clipping with: ', aoiPath, '\n'



						# SAGA-TOOLS start here
						# import as sgrd
						os.system("saga_cmd io_gdal 0 -GRIDS %s -FILES %s -SELECTION 0 -TRANSFORM 0" % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+band))
						
						# resample resBands
						# read grid system from band nr 3
						if fn.endswith(resBands):
							#bandKey = fn[-3:]
							os.system('saga_cmd grid_tools 0 -INPUT %s -OUTPUT %s -SCALE_UP 5 -TARGET_DEFINITION 1 -TARGET_TEMPLATE %s' % (pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+'TMP.sgrd', pathToScenes+tile+'/'+imgFolder+'/'+ fn[:-3] + 'B03' +'.jp2'))
						
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