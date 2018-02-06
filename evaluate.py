import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import threading
import util

print(util.toYellow("======================================================="))
print(util.toYellow("evaluate.py (evaluate/generate point cloud)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data,graph,transform
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=False)
opt.batchSize = opt.inputViewN
opt.chunkSize = 50

# create directories for evaluation output
util.mkdir("results_{0}/{1}".format(opt.group,opt.load))

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	inputImage = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.inH,opt.inW,3])
	renderTrans = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,4])
	depthGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,opt.H,opt.W,1])
	maskGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,opt.H,opt.W,1])
	PH = [inputImage,renderTrans,depthGT,maskGT]
	# ------ build encoder-decoder ------
	encoder = graph.encoder if opt.arch=="original" else \
			  graph.encoder_resnet if opt.arch=="resnet" else None
	decoder = graph.decoder if opt.arch=="original" else \
			  graph.decoder_resnet if opt.arch=="resnet" else None
	latent = encoder(opt,inputImage)
	XYZ,maskLogit = decoder(opt,latent) # [B,H,W,3V],[B,H,W,V]
	mask = tf.to_float(maskLogit>0)
	# ------ build transformer ------
	fuseTrans = tf.nn.l2_normalize(opt.fuseTrans,dim=1)
	XYZid,ML = transform.fuse3D(opt,XYZ,maskLogit,fuseTrans) # [B,1,VHW]

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt,loadNovel=False,loadTest=True)
CADN = len(dataloader.CADs)
chunkN = int(np.ceil(CADN/opt.chunkSize))
dataloader.loadChunk(opt,loadRange=[0,opt.chunkSize])

# prepare model saver/summary writer
saver = tf.train.Saver()

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	util.restoreModel(opt,sess,saver)
	print(util.toMagenta("loading pretrained ({0})...".format(opt.load)))

	for c in range(chunkN):
		dataloader.shipChunk()
		idx = np.arange(c*opt.chunkSize,min((c+1)*opt.chunkSize,CADN))
		if c!=chunkN-1:
			dataloader.thread = threading.Thread(target=dataloader.loadChunk,
												 args=[opt,[(c+1)*opt.chunkSize,min((c+2)*opt.chunkSize,CADN)]])
			dataloader.thread.start()
		dataChunk = dataloader.readyChunk
		testN = len(dataChunk["image_in"])
		# for each CAD model in data chunk
		for i in range(testN):
			m = idx[i]
			CAD = dataloader.CADs[m]
			points24 = np.zeros([opt.inputViewN,1],dtype=np.object)
			# make test batch
			batch = { inputImage: dataChunk["image_in"][i] }
			# evaluate network
			runList = [XYZid,ML]
			xyz,ml = sess.run(runList,feed_dict=batch)
			for a in range(opt.inputViewN):
				xyz1 = xyz[a].T # [VHW,3]
				ml1 = ml[a].reshape([-1]) # [VHW]
				points24[a,0] = xyz1[ml1>0]
			# output results
			scipy.io.savemat("results_{0}/{1}/{2}.mat".format(opt.group,opt.load,CAD),{
				"image": dataChunk["image_in"][i],
				"pointcloud": points24
			})
			pointMeanN = np.array([len(p) for p in points24[:,0]]).mean()
			print("{0}/{1} ({2}) done (average {3} points), time={4}"
				.format(util.toCyan("{0}".format(m+1)),
						util.toCyan("{0}".format(CADN)),
						CAD,
						util.toBlue("{0:.2f}".format(pointMeanN)),
						util.toGreen("{0:.2f}".format(time.time()-timeStart))))
		
		if c!=chunkN-1: dataloader.thread.join()

print(util.toYellow("======= EVALUATION DONE ======="))
