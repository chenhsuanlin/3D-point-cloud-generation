import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import threading
import util

print(util.toYellow("======================================================="))
print(util.toYellow("pretrain.py (pretrain structure generator with fixed viewpoints)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data,graph
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=True)

# create directories for model output
util.mkdir("models_{0}".format(opt.group))

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	inputImage = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.inH,opt.inW,3])
	depthGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.outH,opt.outW,opt.outViewN])
	maskGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.outH,opt.outW,opt.outViewN])
	PH = [inputImage,depthGT,maskGT]
	# ------ build encoder-decoder ------
	encoder = graph.encoder if opt.arch=="original" else \
			  graph.encoder_resnet if opt.arch=="resnet" else None
	decoder = graph.decoder if opt.arch=="original" else \
			  graph.decoder_resnet if opt.arch=="resnet" else None
	latent = encoder(opt,inputImage)
	XYZ,maskLogit = decoder(opt,latent) # [B,H,W,3V],[B,H,W,V]
	depth = XYZ[:,:,:,opt.outViewN*2:opt.outViewN*3]
	mask = tf.to_float(maskLogit>0)
	# ------ define loss ------
	XGT,YGT = np.meshgrid(range(opt.outW),range(opt.outH),indexing="xy") # [H,W]
	XGT,YGT = XGT.astype(np.float32),YGT.astype(np.float32)
	XYGT = np.concatenate([np.tile(XGT,[opt.outViewN,1,1]),
						   np.tile(YGT,[opt.outViewN,1,1])],axis=0) # [V,H,W]
	XYGT = np.expand_dims(np.transpose(XYGT,axes=[1,2,0]),axis=0) # [1,H,W,2V]
	XY = XYZ[:,:,:,:opt.outViewN*2]
	loss_XYZ = graph.l1_loss(XY-XYGT)/opt.batchSize
	loss_XYZ += graph.masked_l1_loss(depth-depthGT,maskLogit>0)/opt.batchSize
	loss_mask = graph.cross_entropy_loss(maskLogit,maskGT)/opt.batchSize
	loss = loss_mask+opt.lambdaDepth*loss_XYZ
	# ------ optimizer ------
	lr_PH = tf.placeholder(tf.float32,shape=[])
	optim = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss)
	# ------ generate summaries ------
	summaryImage = [util.imageSummary(opt,"image_RGB",inputImage,opt.inH,opt.inW),
					util.imageSummary(opt,"image_depth/pred",(1-depth)[:,:,:,0:1],opt.outH,opt.outW),
					util.imageSummary(opt,"image_depth/valid",((1-depth)*mask)[:,:,:,0:1],opt.outH,opt.outW),
					util.imageSummary(opt,"image_depth/GT",(1-depthGT)[:,:,:,0:1],opt.outH,opt.outW),
					util.imageSummary(opt,"image_mask",tf.sigmoid(maskLogit[:,:,:,0:1]),opt.outH,opt.outW),
					util.imageSummary(opt,"image_mask/GT",maskGT[:,:,:,0:1],opt.outH,opt.outW)]
	summaryImage = tf.summary.merge(summaryImage)
	summaryLoss = [tf.summary.scalar("loss_total",loss),
				   tf.summary.scalar("loss_mask",loss_mask),
				   tf.summary.scalar("loss_XYZ",loss_XYZ)]
	summaryLoss = tf.summary.merge(summaryLoss)

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt,loadNovel=False,loadFixedOut=True)
dataloader.loadChunk(opt)

# prepare model saver/summary writer
saver = tf.train.Saver(max_to_keep=50)
summaryWriter = tf.summary.FileWriter("summary_{0}/{1}".format(opt.group,opt.model))

print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	if opt.fromIt!=0:
		util.restoreModelFromIt(opt,sess,saver,opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	else:
		summaryWriter.add_graph(sess.graph)
	print(util.toMagenta("start training..."))

	chunkResumeN,chunkMaxN = opt.fromIt//opt.itPerChunk,opt.toIt//opt.itPerChunk
	# training loop
	for c in range(chunkResumeN,chunkMaxN):
		dataloader.shipChunk()
		dataloader.thread = threading.Thread(target=dataloader.loadChunk,args=[opt])
		dataloader.thread.start()
		for i in range(c*opt.itPerChunk,(c+1)*opt.itPerChunk):
			lr = opt.lr*opt.lrDecay**(i//opt.lrStep)
			# make training batch
			batch = data.makeBatchFixed(opt,dataloader,PH)
			batch[lr_PH] = lr
			# run one step
			runList = [optim,loss,loss_XYZ,loss_mask,maskLogit]
			_,l,lx,lm,ml = sess.run(runList,feed_dict=batch)
			if (i+1)%50==0:
				print("it. {0}/{1}, lr={2}, loss={4} ({5},{6}), time={3}"
					.format(util.toCyan("{0}".format(i+1)),
							opt.toIt,
							util.toYellow("{0:.0e}".format(lr)),
							util.toGreen("{0:.2f}".format(time.time()-timeStart)),
							util.toRed("{0:.2f}".format(l)),
							util.toRed("{0:.2f}".format(lx)),
							util.toRed("{0:.2f}".format(lm))))
			if (i+1)%200==0:
				summaryWriter.add_summary(sess.run(summaryLoss,feed_dict=batch),i+1)
			if (i+1)%1000==0:
				summaryWriter.add_summary(sess.run(summaryImage,feed_dict=batch),i+1)
			if (i+1)%10000==0:
				util.saveModel(opt,sess,saver,i+1)
				print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.model,i+1)))
		dataloader.thread.join()

print(util.toYellow("======= TRAINING DONE ======="))
