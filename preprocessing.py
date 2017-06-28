############################
###### IMPORT TOOLS ########
############################

import sys
sys.path.append('/run/media/csaq7453/Elements/VU_Fernerkundung/Projekt/tools/')

from library import *

############################
###### FOLDER STRUCTURE ####
############################
# make sure each directory ends with a slash

# change for windows/linux, university/home os
drive = '/run/media/csaq7453/Elements/'
#drive = 'E:/'

projectFolder = drive + 'VU_Fernerkundung/Projekt/'

sentinelFolder = drive + 'S2/'
sentinelOut = drive + 'tiles/'

ex_tile = sentinelOut + 'S2A_OPER_MSI_L1C_TL_SGS__20160827T153533_A006168_T32TPT_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160827T153533_A006168_T32TPT_B03.tif'

# Area of Interest
aoiShapefile = drive + projectFolder + 'daten_Ladner/FSH_AOI.shp'


# ground-truth (gt)-data, vector files
gtShapes = drive + 'VU_Fernerkundung/Projekt/input_vector/'
# Directory for rasterized gt-Data
gtRasters = drive + 'VU_Fernerkundung/Projekt/input_rasterized/'


############################
###### MAIN ################
############################


extent = getAOI(aoiShapefile)

findImagery(sentinelFolder, shapefileBoundaries = extent, outFolder = sentinelOut)

metaClippedScene = clipScenes(sentinelOut, aoiShapefile)


cols, rows, geo_transform, projection = getMeta(ex_tile)

labelByValue = rasterizeVectorData(gtShapes, gtRasters, cols, rows, geo_transform, projection)