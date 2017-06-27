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
drive = '/run/media/csaq7453/Elements/'
#drive = 'E:/'

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

'''
extent = getAOI(aoiShapefile)

findImagery(sentinelFolder, shapefileBoundaries = extent, outFolder = sentinelOut)

metaClippedScene = clipScenes(sentinelOut, aoiShapefile)


cols, rows, geo_transform, projection = getMeta(ex_tile)

labelByValue = rasterizeVectorData(gtShapes, gtRasters, cols, rows, geo_transform, projection)
'''

cols, rows, geo_transform, projection = getMeta(ex_tile)

labeled_pixels, labelByIndex = loadRasters(gtRasters)

labelByValue = getClassValue(gtRasters)

S2Data = loadS2(sentinelOut, cols, rows)




#sigmaList=[0.15,0.2,0.4,0.6,0.8,1.0]
cList=[1.0]
sigmaList=[1.0]





'''

# define trainPixSize (per class)
#nrSamples=[100,250,500,1000, 2000, 5000, 10000]
nrSamples = [10,20,30,40,50]

timeClf = np.zeros((len(nrSamples)))
timePred = np.zeros((len(nrSamples)))

# create file to track processing timefobj = open("%sprocessing_time.txt"%(projectFolder), "w")
fobj.write('trainingPixels\ttimeToClassify\ttimeToPredict\n')


for i, count in enumerate(nrSamples):
	timeClf[i], timePred[i] = wrapSVM(S2Data, projectFolder, cols, rows, geo_transform, projection, labeled_pixels, count, labelByValue)
	fobj.write('%i\t%f\t%f\n'%(count,timeClf[i], timePred[i]))
	

fobj.close()



pixNumArray = np.asarray(nrSamples)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(0,60)
ax.plot(pixNumArray, timeClf, 'g--', pixNumArray, timePred, 'r-.')
ax.set_xlabel('Samplesize in Nr. of Pixels')
ax.set_ylabel('Time [sec]')
ax.set_title('Time needed to train the classifier vs.\n time needed to predict the entire image')
#ax.legend(loc='upper left')
plt.show()
'''