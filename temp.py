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

				# clean *.mgrd files	
				os.remove(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.mgrd')
				
				# delete TMP.sgrds
				print 'Delete files: ', pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat', '\n'
				os.system('gdalmanage delete %s'%(pathToScenes+tile+'/'+imgFolder+'/'+ 'TMP.sdat'))




############################
###### FOLDER STRUCTURE ####
############################
# make sure each directory ends with a slash
# --> avoid trouble when joining pathnames and filenames, --> fix ...

# change for windows/linux, university/home os
drive = '/run/media/csaq7453/Elements/'


sentinelFolder = drive + 'S2/'
sentinelOut = drive + 'tiles/'

ex_tile = drive + 'S2/20160628/S2A_OPER_PRD_MSIL1C_PDMC_20160628T172504_R022_V20160628T101026_20160628T101026.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160628T153712_A005310_T32TPT_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160628T153712_A005310_T32TPT_B04.jp2'
ex_shape = drive + 'VU_Fernerkundung/Projekt/input_vector/Wald_4.shp'

# Area of Interest
aoiShapefile = drive + 'VU_Fernerkundung/Projekt/daten_Ladner/FSH_AOI.shp'


# ground-truth (gt)-data, vector files and rasterized
gtShapes = drive + 'VU_Fernerkundung/Projekt/input_vector/'
# Directory for rasterized gt
gtRasters = drive + 'VU_Fernerkundung/Projekt/input_rasterized/'




extent = getAOI(aoiShapefile)

findImagery(sentinelFolder, shapefileBoundaries = extent, outFolder = sentinelOut)

clipScenes(sentinelOut, aoiShapefile)

