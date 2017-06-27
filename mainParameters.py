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
# --> avoid trouble when joining pathnames and filenames, --> fix ...

# change for windows/linux, university/home os
#drive = '/run/media/csaq7453/Elements/'
drive = 'E:/'

projectFolder = drive + 'VU_Fernerkundung/Projekt/'


sentinelFolder = drive + 'S2/'
sentinelOut = drive + 'tiles/'

ex_tile = sentinelOut + 'S2A_OPER_MSI_L1C_TL_SGS__20160827T153533_A006168_T32TPT_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160827T153533_A006168_T32TPT_B03.tif'

# Area of Interest
aoiShapefile = drive + 'VU_Fernerkundung/Projekt/daten_Ladner/FSH_AOI.shp'


# ground-truth (gt)-data, vector files and rasterized
gtShapes = drive + 'VU_Fernerkundung/Projekt/input_vector/'
# Directory for rasterized gt
gtRasters = drive + 'VU_Fernerkundung/Projekt/input_rasterized/'

############################
###### MAIN ################
############################


from osgeo import gdal
import os
import numpy as np


cols, rows, geo_transform, projection = getMeta(ex_tile)

labeled_pixels, labelByIndex = loadRasters(gtRasters)

labelByValue = getClassValue(gtRasters)

S2Data = loadS2(sentinelOut, cols, rows)



#cList=[0.2,0.4,0.6,0.8,1.0, 2.0]
cList = [1.0]

svmParam(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, labelByValue, cList)

'''
from matplotlib import pyplot as plt
f = plt.figure()
f.add_subplot(1, 2, 2)
r = S2Data[:,:,2]
g = S2Data[:,:,1]
b = S2Data[:,:,0]
rgb = np.dstack([r,g,b])
f.add_subplot(1, 2, 1)
plt.imshow(rgb/255)
f.add_subplot(1, 2, 2)
plt.imshow(classification)
'''