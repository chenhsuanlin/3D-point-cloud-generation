import numpy as np
import scipy.linalg
import scipy.io
import os,time
import threading
import tensorflow as tf
import util

class Loader():
	def __init__(self,opt,loadNovel=True,loadFixedOut=False,loadTest=False):
		self.loadNovel = loadNovel
		self.loadFixedOut = loadFixedOut
		self.loadTest = loadTest
		listFile = "data/{0}_{1}.list".format(opt.category,"test" if loadTest else "train")
		self.CADs = []
		with open(listFile) as file:
			for line in file:
				id = line.strip().split("/")[1]
				self.CADs.append(id)
		self.CADs.sort()
	def loadChunk(self,opt,loadRange=None):
		data = {}
		if loadRange is not None: idx = np.arange(loadRange[0],loadRange[1])
		else: idx = np.random.permutation(len(self.CADs))[:opt.chunkSize]
		chunkSize = len(idx)
		# preallocate memory
		data["image_in"] = np.ones([chunkSize,24,opt.inH,opt.inW,3],dtype=np.float32)
		if self.loadNovel:
			data["depth"] = np.ones([chunkSize,opt.sampleN,opt.H,opt.W],dtype=np.float32)
			data["mask"] = np.ones([chunkSize,opt.sampleN,opt.H,opt.W],dtype=np.bool)
			data["trans"] = np.ones([chunkSize,opt.sampleN,4],dtype=np.float32)
		if self.loadFixedOut:
			data["depth_fixedOut"] = np.ones([chunkSize,opt.outViewN,opt.outH,opt.outW],dtype=np.float32)
			data["mask_fixedOut"] = np.ones([chunkSize,opt.outViewN,opt.outH,opt.outW],dtype=np.bool)
		# load data
		for c in range(chunkSize):
			CAD = self.CADs[idx[c]]
			data["image_in"][c] = np.load("data/{0}_inputRGB/{1}.npy".format(opt.category,CAD))/255.0
			if self.loadNovel:
				rawData = scipy.io.loadmat("data/{0}_depth/{1}.mat".format(opt.category,CAD))
				depth = rawData["Z"]
				trans = rawData["trans"]
				mask = depth!=0
				depth[~mask] = opt.renderDepth
				# store data
				data["depth"][c] = depth
				data["mask"][c] = mask
				data["trans"][c] = trans
			if self.loadFixedOut:
				rawData_fixed = scipy.io.loadmat("data/{0}_depth_fixed{1}/{2}.mat".format(opt.category,opt.outViewN,CAD))
				depth_fixed = rawData_fixed["Z"]
				mask_fixed = depth_fixed!=0
				depth_fixed[~mask_fixed] = opt.renderDepth
				# store data
				data["depth_fixedOut"][c] = depth_fixed
				data["mask_fixedOut"][c] = mask_fixed
		self.pendingChunk = data
	def shipChunk(self):
		self.readyChunk,self.pendingChunk = self.pendingChunk,None

# make training batch
def makeBatch(opt,dataloader,PH):
	data = dataloader.readyChunk
	inputImage,targetTrans,depthGT,maskGT = PH
	modelIdx = np.random.permutation(opt.chunkSize)[:opt.batchSize]
	modelIdxTile = np.tile(modelIdx,[opt.novelN,1]).T
	angleIdx = np.random.randint(24,size=[opt.batchSize])
	sampleIdx = np.random.randint(opt.sampleN,size=[opt.batchSize,opt.novelN])
	batch = {
		inputImage: data["image_in"][modelIdx,angleIdx],
		targetTrans: data["trans"][modelIdxTile,sampleIdx],
		depthGT: np.expand_dims(data["depth"][modelIdxTile,sampleIdx],axis=-1),
		maskGT: np.expand_dims(data["mask"][modelIdxTile,sampleIdx],axis=-1)
	}
	return batch

# make training batch
def makeBatchFixed(opt,dataloader,PH):
	data = dataloader.readyChunk
	inputImage,depthGT,maskGT = PH
	modelIdx = np.random.permutation(opt.chunkSize)[:opt.batchSize]
	angleIdx = np.random.randint(24,size=[opt.batchSize])
	batch = {
		inputImage: data["image_in"][modelIdx,angleIdx],
		depthGT: np.transpose(data["depth_fixedOut"][modelIdx],axes=[0,2,3,1]),
		maskGT: np.transpose(data["mask_fixedOut"][modelIdx],axes=[0,2,3,1])
	}
	return batch
