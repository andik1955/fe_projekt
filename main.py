from library import findImagery





############################
###### MAIN ################
############################

sentinelFolder = 'C:/S2/'
sentinelOut = 'C:/imagery/'

aoiShapefile = ''
groundtruthShapes = ''


# get extent of AOI
extent = getAOI(aoiShapefile)

# find imagery/tiles that contain AOI and save them to outFolder --> error when input folder is specified
findImagery(sentinelFolder, outFolder='C:/S2/') # specify boundary and outFolder --> sentinelOut

# clip those remaining tiles to AOI and retrieve raster metadata
ncols, nrows, geo_transform, projection = clipScenes(sentinelOut, aoiShapefile)

# 
createTrainingValidation(gt_array, ratioTrainValid, rows, cols, geo_transform, projection):