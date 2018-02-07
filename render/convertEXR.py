import os,sys,time
import numpy as np
import scipy.io
import OpenEXR
import array,Imath

CATEGORY = sys.argv[-4]
MODEL_LIST = sys.argv[-3]
RESOLUTION = int(sys.argv[-2])
FIXED = int(sys.argv[-1])
N = 100

def readEXR(fname,RESOLUTION):
	channel_list = ["B","G","R"]
	file = OpenEXR.InputFile(fname)
	dw = file.header()["dataWindow"]
	height,width = RESOLUTION,RESOLUTION
	FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
	vectors = [np.array(array.array("f",file.channel(c,FLOAT))) for c in channel_list]
	depth = vectors[0].reshape([height,width])
	return depth

listFile = open(MODEL_LIST)
for line in listFile:
	MODEL = line.strip()
	timeStart = time.time()

	# arbitrary views
	Z = []
	depth_path = "output/{1}_depth/exr_{0}".format(MODEL,CATEGORY)
	for i in range(N):
		depth = readEXR("{0}/{1}.exr".format(depth_path,i),RESOLUTION)
		depth[np.isinf(depth)] = 0
		Z.append(depth)
	trans_path = "{0}/trans.mat".format(depth_path)
	trans = scipy.io.loadmat(trans_path)["trans"]
	mat_path = "output/{1}_depth/{0}.mat".format(MODEL,CATEGORY)
	scipy.io.savemat(mat_path,{
		"Z": np.stack(Z),
		"trans": trans,
	})
	os.system("rm -rf {0}".format(depth_path))

	# fixed views
	Z = []
	depth_path = "output/{1}_depth_fixed{2}/exr_{0}".format(MODEL,CATEGORY,FIXED)
	for i in range(FIXED):
		depth = readEXR("{0}/{1}.exr".format(depth_path,i),RESOLUTION)
		depth[np.isinf(depth)] = 0
		Z.append(depth)
	mat_path = "output/{1}_depth_fixed{2}/{0}.mat".format(MODEL,CATEGORY,FIXED)
	scipy.io.savemat(mat_path,{
		"Z": np.stack(Z),
	})
	os.system("rm -rf {0}".format(depth_path))

	print("{1} done, time={0:.4f} sec".format(time.time()-timeStart,MODEL))
