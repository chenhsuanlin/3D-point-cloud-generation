import numpy as np
import argparse,os
import scipy.linalg
import tensorflow as tf
import util

def set(training):

	# parse input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--category",				default="03001627",	help="category ID number")
	parser.add_argument("--group",					default="0",		help="name for group")
	parser.add_argument("--model",					default="test",		help="name for model instance")
	parser.add_argument("--load",					default=None,		help="load trained model to fine-tune/evaluate")
	parser.add_argument("--std",		type=float,	default=0.1,		help="initialization standard deviation")
	parser.add_argument("--outViewN",	type=int,	default=8,			help="number of fixed views (output)")
	parser.add_argument("--inSize",					default="64x64",	help="resolution of encoder input")
	parser.add_argument("--outSize",				default="128x128",	help="resolution of decoder output")
	parser.add_argument("--predSize",				default="128x128",	help="resolution of prediction")
	parser.add_argument("--upscale",	type=int,	default=5,			help="upscaling factor for rendering")
	parser.add_argument("--novelN",		type=int,	default=5,			help="number of novel views simultaneously")
	parser.add_argument("--arch",					default=None)
	if training: # training
		parser.add_argument("--batchSize",	type=int,	default=20,			help="batch size for training")
		parser.add_argument("--chunkSize",	type=int,	default=100,		help="data chunk size to load")
		parser.add_argument("--itPerChunk",	type=int,	default=50,			help="training iterations per chunk")
		parser.add_argument("--lr",			type=float,	default=1e-4,		help="base learning rate (AE)")
		parser.add_argument("--lrDecay",	type=float,	default=1.0,		help="learning rate decay multiplier")
		parser.add_argument("--lrStep",		type=int,	default=20000,		help="learning rate decay step size")
		parser.add_argument("--lambdaDepth",type=float,	default=1.0,		help="loss weight factor (depth)")
		parser.add_argument("--fromIt",		type=int,	default=0,			help="resume training from iteration number")
		parser.add_argument("--toIt",		type=int,	default=100000,		help="run training to iteration number")
	else: # evaluation
		parser.add_argument("--batchSize",	type=int,	default=1,		help="batch size for evaluation")
	opt = parser.parse_args()

	# these stay fixed
	opt.sampleN = 100
	opt.renderDepth = 1.0
	opt.BNepsilon = 1e-5
	opt.BNdecay = 0.999
	opt.inputViewN = 24
	# ------ below automatically set ------
	opt.training = training
	opt.inH,opt.inW = [int(x) for x in opt.inSize.split("x")]
	opt.outH,opt.outW = [int(x) for x in opt.outSize.split("x")]
	opt.H,opt.W = [int(x) for x in opt.predSize.split("x")]
	opt.visBlockSize = int(np.floor(np.sqrt(opt.batchSize)))
	opt.Khom3Dto2D = np.array([[opt.W,0 ,0,opt.W/2],
							   [0,-opt.H,0,opt.H/2],
							   [0,0,-1,0],
							   [0,0, 0,1]],dtype=np.float32)
	opt.Khom2Dto3D = np.array([[opt.outW,0 ,0,opt.outW/2],
							   [0,-opt.outH,0,opt.outH/2],
							   [0,0,-1,0],
							   [0,0, 0,1]],dtype=np.float32)
	opt.fuseTrans = np.load("trans_fuse{0}.npy".format(opt.outViewN))

	print("({0}) {1}".format(
		util.toGreen("{0}".format(opt.group)),
		util.toGreen("{0}".format(opt.model))))
	print("------------------------------------------")
	print("batch size: {0}, category: {1}".format(
		util.toYellow("{0}".format(opt.batchSize)),
		util.toYellow("{0}".format(opt.category))))
	print("size: {0}x{1}(in), {2}x{3}(out), {4}x{5}(pred)".format(
		util.toYellow("{0}".format(opt.inH)),
		util.toYellow("{0}".format(opt.inW)),
		util.toYellow("{0}".format(opt.outH)),
		util.toYellow("{0}".format(opt.outW)),
		util.toYellow("{0}".format(opt.H)),
		util.toYellow("{0}".format(opt.W))))
	if training:
		print("learning rate: {0} (decay: {1}, step size: {2})".format(
			util.toYellow("{0:.2e}".format(opt.lr)),
			util.toYellow("{0}".format(opt.lrDecay)),
			util.toYellow("{0}".format(opt.lrStep))))
		print("depth loss weight: {0}".format(
			util.toYellow("{0}".format(opt.lambdaDepth))))
	print("viewN: {0}(out), upscale: {1}, novelN: {2}".format(
		util.toYellow("{0}".format(opt.outViewN)),
		util.toYellow("{0}".format(opt.upscale)),
		util.toYellow("{0}".format(opt.novelN))))
	print("------------------------------------------")
	if training:
		print(util.toMagenta("training model ({0}) {1}...".format(opt.group,opt.model)))

	return opt
