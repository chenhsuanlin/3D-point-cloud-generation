import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import threading
import util

print(util.toYellow("======================================================="))
print(util.toYellow("evaluate_dist.py (evaluate average distance of generated point cloud)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=False)

with tf.device("/gpu:0"):
	VsPH = tf.placeholder(tf.float64,[None,3])
	VtPH = tf.placeholder(tf.float64,[None,3])
	_,minDist = util.projection(VsPH,VtPH)

# compute test error for one prediction
def computeTestError(Vs,Vt,type):
	VsN,VtN = len(Vs),len(Vt)
	if type=="pred->GT": evalN,VsBatchSize,VtBatchSize = min(VsN,200),200,100000
	if type=="GT->pred": evalN,VsBatchSize,VtBatchSize = min(VsN,200),200,40000
	# randomly sample 3D points to evaluate (for speed)
	randIdx = np.random.permutation(VsN)[:evalN]
	Vs_eval = Vs[randIdx]
	minDist_eval = np.ones([evalN])*np.inf
	# for batches of source vertices
	VsBatchN = int(np.ceil(evalN/VsBatchSize))
	VtBatchN = int(np.ceil(VtN/VtBatchSize))
	for b in range(VsBatchN):
		VsBatch = Vs_eval[b*VsBatchSize:(b+1)*VsBatchSize]
		minDist_batch = np.ones([len(VsBatch)])*np.inf
		for b2 in range(VtBatchN):
			VtBatch = Vt[b2*VtBatchSize:(b2+1)*VtBatchSize]
			md = sess.run(minDist,feed_dict={ VsPH:VsBatch, VtPH:VtBatch })
			minDist_batch = np.minimum(minDist_batch,md)
		minDist_eval[b*VsBatchSize:(b+1)*VsBatchSize] = minDist_batch
	return np.mean(minDist_eval)

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt,loadNovel=False,loadTest=True)
CADN = len(dataloader.CADs)

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	pred2GT_all = np.ones([CADN,opt.inputViewN])*np.inf
	GT2pred_all = np.ones([CADN,opt.inputViewN])*np.inf

	for m in range(CADN):
		CAD = dataloader.CADs[m]
		# load GT
		obj = scipy.io.loadmat("data/{0}_testGT/{1}.mat".format(opt.category,CAD))
		Vgt = np.concatenate([obj["V"],obj["Vd"]],axis=0)
		VgtN = len(Vgt)
		# load prediction
		Vpred24 = scipy.io.loadmat("results_{0}/{1}/{2}.mat".format(opt.group,opt.load,CAD))["pointcloud"][:,0]
		assert(len(Vpred24)==opt.inputViewN)

		for a in range(opt.inputViewN):
			Vpred = Vpred24[a]
			VpredN = len(Vpred)
			# rotate CAD model to be in consistent coordinates
			Vpred[:,1],Vpred[:,2] = Vpred[:,2],-Vpred[:,1]
			# compute test error in both directions
			pred2GT_all[m,a] = computeTestError(Vpred,Vgt,type="pred->GT")
			GT2pred_all[m,a] = computeTestError(Vgt,Vpred,type="GT->pred")

		print("{0}/{1} {2}: {3}(pred->GT),{4}(GT->pred), time={5}"
			.format(util.toCyan("{0}".format(m+1)),
					util.toCyan("{0}".format(CADN)),
					CAD,
					util.toRed("{0:.4f}".format(pred2GT_all[m].mean()*100)),
					util.toRed("{0:.4f}".format(GT2pred_all[m].mean()*100)),
					util.toGreen("{0:.2f}".format(time.time()-timeStart))))

	scipy.io.savemat("results_{0}/{1}/testerror.mat".format(opt.group,opt.load),{
		"pred2GT": pred2GT_all,
		"GT2pred": GT2pred_all
	})

print(util.toYellow("======= EVALUATION DONE ======="))
