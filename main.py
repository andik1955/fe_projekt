############################
###### MAIN ################
############################

# make sure each directory ends with a slash
# --> avoid trouble when joining pathnames and filenames, --> fix ...

# change for windows/linux, university/home os
drive = '/run/media/csaq7453/Elements/'


sentinelFolder = drive + 'S2/'
sentinelOut = drive + 'imagery/'

ex_tile = drive + 'S2/20160628/S2A_OPER_PRD_MSIL1C_PDMC_20160628T172504_R022_V20160628T101026_20160628T101026.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160628T153712_A005310_T32TPT_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160628T153712_A005310_T32TPT_B04.jp2'
ex_shape = drive + 'VU_Fernerkundung/Projekt/input_vector/Wald_4.shp'



# Area of Interest
aoiShapefile = drive + 'VU_Fernerkundung/Projekt/Ladner_processed/FSH_AOI.shp'


# ground-truth (gt)-data, vector files and rasterized
gtShapes = drive + 'VU_Fernerkundung/Projekt/input_vector/'
# Directory for rasterized gt
gtRasters = drive + 'VU_Fernerkundung/Projekt/input_rasterized/'


'''
rows, cols, geo_transform, projection = getMeta(ex_tile)

print 'Img Metadata: ', geo_transform
'''


'''
gtList, label_spec = rasterizeVectorData(gtShapes, gtRasters, cols, rows, geo_transform, projection, target_value = 1)


print 2*'\n', 'Rasterized Data\n', 25*'-', '\n'

for el in gtList:
	print el, '\n'

#print type(ds)

#already done!
'''

# get gt data read gt data as numpy array







'''
# get extent of AOI
extent = getAOI(aoiShapefile)


# find imagery/tiles that contain AOI and save them to outFolder --> error when input folder is specified
findImagery(sentinelFolder, shapefileBoundaries = extent, outFolder = sentinelOut)


# clip those remaining tiles to AOI and retrieve raster metadata
ncols, nrows, geo_transform, projection = clipScenes(sentinelOut, aoiShapefile)

# rasterize ground truth data

create_mask_from_vector(gtShapes, gtRasters, cols, rows, geo_transform, projection, target_value = 1)

# rasterize data here


'''