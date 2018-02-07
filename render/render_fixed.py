import os,sys,time
import bpy
import numpy as np
import shutil
import scipy.io

curpath = os.path.abspath(os.path.dirname("."))
sys.path.insert(0,curpath)
import util

# usage: blender blank.blend -b -P render_fixed.py -- SHAPENETPATH CATEGORY MODEL_LIST RESOLUTION FIXED

# redirect output to log file
logfile = "/tmp/blender_render.log"

SHAPENETPATH = sys.argv[-5]
CATEGORY = sys.argv[-4]
MODEL_LIST = sys.argv[-3]
RESOLUTION = int(sys.argv[-2])
FIXED = int(sys.argv[-1])

scene,camera,fo = util.setupBlender("buffer_fixed",RESOLUTION)
camPosAll = util.getFixedViews(FIXED)

listFile = open(MODEL_LIST)
for line in listFile:
	MODEL = line.strip()
	timeStart = time.time()
	trans = []

	depth_path = "output/{1}_depth_fixed{2}/exr_{0}".format(MODEL,CATEGORY,FIXED)
	if not os.path.isdir(depth_path):
		os.makedirs(depth_path)

	# suppress output
	open(logfile,"a").close()
	old = os.dup(1)
	sys.stdout.flush()
	os.close(1)
	os.open(logfile,os.O_WRONLY)

	shape_file = "{2}/{0}/{1}/models/model_normalized.obj".format(CATEGORY,MODEL,SHAPENETPATH)
	bpy.ops.import_scene.obj(filepath=shape_file)

	for m in bpy.data.materials:
		m.use_shadeless = True

	for i in range(FIXED):
		theta = 0
		camPos = camPosAll[i]
		q1 = util.camPosToQuaternion(camPos)
		q2 = util.camRotQuaternion(camPos,theta)
		q = util.quaternionProduct(q2,q1)

		util.setCameraExtrinsics(camera,camPos,q)
		q_extr,t_extr = util.cameraExtrinsicMatrix(q,camPos)

		# for ShapeNetCore.v2 all the objects are rotated 90 degrees
		# comment out this block if ShapeNetCore.v1 is used
		if i==0:
			for o in bpy.data.objects:
				if o==camera: o.select = False
				else: o.select = True
			bpy.ops.transform.rotate(value=-np.pi/2,axis=(0,0,1))

		bpy.ops.render.render(write_still=False)

		shutil.copyfile("{0}/Z0001.exr".format(fo.base_path),
						"{0}/{1}.exr".format(depth_path,i))
		trans.append(np.array(q_extr))

	# show output
	os.close(1)
	os.dup(old)
	os.close(old)

	# clean up
	for o in bpy.data.objects:
		if o==camera: continue
		o.select = True
	bpy.ops.object.delete()
	for m in bpy.data.meshes:
		bpy.data.meshes.remove(m)
	for m in bpy.data.materials:
	    m.user_clear()
	    bpy.data.materials.remove(m)

	print("{1} done, time={0:.4f} sec".format(time.time()-timeStart,MODEL))

trans = np.array(trans,dtype=np.float32)
np.save("output/trans_fuse{0}.npy".format(FIXED),trans)
