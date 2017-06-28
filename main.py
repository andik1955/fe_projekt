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

## Load data

import numpy as np

cols, rows, geo_transform, projection = getMeta(ex_tile)

labeled_pixels, labelByIndex = loadRasters(gtRasters)

labelByValue = getClassValue(gtRasters)

S2Data = loadS2(sentinelOut, cols, rows)

## Classification runs to test different values of C
# define values for c
cList = np.round(np.geomspace(start = 0.2, stop = 5.0, num = 20), decimals=2)

svmParam(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, 5000, labelByValue, cList)


## Classification runs to test different TrainingSampleSizes

# define trainPixSize (per class)
nrSamples=[50, 100, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]

wrapSVM(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, nrSamples, labelByValue)