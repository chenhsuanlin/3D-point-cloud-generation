import numpy as np
import tensorflow as tf
import time

# build transformer (3D generator)
def fuse3D(opt,XYZ,maskLogit,fuseTrans): # [B,H,W,3V],[B,H,W,V]
	with tf.name_scope("transform_fuse3D"):
		XYZ = tf.transpose(XYZ,perm=[0,3,1,2]) # [B,3V,H,W]
		maskLogit = tf.transpose(maskLogit,perm=[0,3,1,2]) # [B,V,H,W]
		# 2D to 3D coordinate transformation
		invKhom = np.linalg.inv(opt.Khom2Dto3D)
		invKhomTile = np.tile(invKhom,[opt.batchSize,opt.outViewN,1,1])
		# viewpoint rigid transformation
		q_view = fuseTrans
		t_view = np.tile([0,0,-opt.renderDepth],[opt.outViewN,1]).astype(np.float32)
		RtHom_view = transParamsToHomMatrix(q_view,t_view)
		RtHomTile_view = tf.tile(tf.expand_dims(RtHom_view,0),[opt.batchSize,1,1,1])
		invRtHomTile_view = tf.matrix_inverse(RtHomTile_view)
		# effective transformation
		RtHomTile = tf.matmul(invRtHomTile_view,invKhomTile) # [B,V,4,4]
		RtTile = RtHomTile[:,:,:3,:] # [B,V,3,4]
		# transform depth stack
		ML = tf.reshape(maskLogit,[opt.batchSize,1,-1]) # [B,1,VHW]
		XYZhom = get3DhomCoord(XYZ,opt) # [B,V,4,HW]
		XYZid = tf.matmul(RtTile,XYZhom) # [B,V,3,HW]
		# fuse point clouds
		XYZid = tf.reshape(tf.transpose(XYZid,perm=[0,2,1,3]),[opt.batchSize,3,-1]) # [B,3,VHW]
	return XYZid,ML # [B,1,VHW]

# build transformer (render 2D depth)
def render2D(opt,XYZid,ML,renderTrans): # [B,1,VHW]
	offsetDepth,offsetMaskLogit = 10.0,1.0
	with tf.name_scope("transform_render2D"):
		# target rigid transformation
		q_target = tf.reshape(renderTrans,[opt.batchSize*opt.novelN,4])
		t_target = np.tile([0,0,-opt.renderDepth],[opt.batchSize*opt.novelN,1]).astype(np.float32)
		RtHom_target = tf.reshape(transParamsToHomMatrix(q_target,t_target),[opt.batchSize,opt.novelN,4,4])
		# 3D to 2D coordinate transformation
		KupHom = opt.Khom3Dto2D*np.array([[opt.upscale],[opt.upscale],[1],[1]],dtype=np.float32)
		KupHomTile = np.tile(KupHom,[opt.batchSize,opt.novelN,1,1])
		# effective transformation
		RtHomTile = tf.matmul(KupHomTile,RtHom_target) # [B,N,4,4]
		RtTile = RtHomTile[:,:,:3,:] # [B,N,3,4]
		# transform depth stack
		XYZidHom = get3DhomCoord2(XYZid,opt) # [B,4,VHW]
		XYZidHomTile = tf.tile(tf.expand_dims(XYZidHom,axis=1),[1,opt.novelN,1,1]) # [B,N,4,VHW]
		XYZnew = tf.matmul(RtTile,XYZidHomTile) # [B,N,3,VHW]
		Xnew,Ynew,Znew = tf.split(XYZnew,3,axis=2) # [B,N,1,VHW]
		# concatenate all viewpoints
		MLcat = tf.reshape(tf.tile(ML,[1,opt.novelN,1]),[-1]) # [BNVHW]
		XnewCat = tf.reshape(Xnew,[-1]) # [BNVHW]
		YnewCat = tf.reshape(Ynew,[-1]) # [BNVHW]
		ZnewCat = tf.reshape(Znew,[-1]) # [BNVHW]
		batchIdxCat,novelIdxCat,_ = np.meshgrid(range(opt.batchSize),range(opt.novelN),range(opt.outViewN*opt.outH*opt.outW),indexing="ij")
		batchIdxCat,novelIdxCat = batchIdxCat.reshape([-1]),novelIdxCat.reshape([-1]) # [BNVHW]
		# apply in-range masks
		XnewCatInt = tf.to_int32(tf.round(XnewCat))
		YnewCatInt = tf.to_int32(tf.round(YnewCat))
		maskInside = (XnewCatInt>=0)&(XnewCatInt<opt.upscale*opt.W)&(YnewCatInt>=0)&(YnewCatInt<opt.upscale*opt.H)
		valueInt = tf.stack([XnewCatInt,YnewCatInt,batchIdxCat,novelIdxCat],axis=1) # [BNVHW,d]
		valueFloat = tf.stack([1/(ZnewCat+offsetDepth+1e-8),MLcat],axis=1) # [BNVHW,d]
		insideInt = tf.boolean_mask(valueInt,maskInside) # [U,d]
		insideFloat = tf.boolean_mask(valueFloat,maskInside) # [U,d]
		_,MLnewValid = tf.unstack(insideFloat,axis=1) # [U]
		# apply visible masks
		maskExist = MLnewValid>0
		visInt = tf.boolean_mask(insideInt,maskExist)
		visFloat = tf.boolean_mask(insideFloat,maskExist)
		invisInt = tf.boolean_mask(insideInt,~maskExist)
		invisFloat = tf.boolean_mask(insideFloat,~maskExist)
		XnewVis,YnewVis,batchIdxVis,novelIdxVis = tf.unstack(visInt,axis=1) # [U]
		iZnewVis,MLnewVis = tf.unstack(visFloat,axis=1) # [U]
		XnewInvis,YnewInvis,batchIdxInvis,novelIdxInvis = tf.unstack(invisInt,axis=1) # [U]
		_,MLnewInvis = tf.unstack(invisFloat,axis=1) # [U]
		# map to upsampled inverse depth and mask (visible)
		scatterIdx = tf.stack([batchIdxVis,novelIdxVis,YnewVis,XnewVis],axis=1) # [U,4]
		scatterShape = tf.constant([opt.batchSize,opt.novelN,opt.H*opt.upscale,opt.W*opt.upscale,3])
		countOnes = tf.ones_like(iZnewVis)
		scatteriZMLCnt = tf.stack([iZnewVis,MLnewVis,countOnes],axis=1) # [U,3]
		upNewiZMLCnt = tf.scatter_nd(scatterIdx,scatteriZMLCnt,scatterShape) # [B,N,uH,uW,3]
		upNewiZMLCnt = tf.reshape(upNewiZMLCnt,[opt.batchSize*opt.novelN,opt.H*opt.upscale,opt.W*opt.upscale,3]) # [BN,uH,uW,3]
		# downsample back to original size
		newiZMLCnt = tf.nn.max_pool(upNewiZMLCnt,ksize=[1,opt.upscale,opt.upscale,1],
												 strides=[1,opt.upscale,opt.upscale,1],padding="VALID") # [BN,H,W,3]
		newiZMLCnt = tf.reshape(newiZMLCnt,[opt.batchSize,opt.novelN,opt.H,opt.W,3]) # [B,N,H,W,3]
		newInvDepth,newMaskLogitVis,collision = tf.split(newiZMLCnt,3,axis=4) # [B,N,H,W,1]
		# map to upsampled inverse depth and mask (invisible)
		scatterIdx = tf.stack([batchIdxInvis,novelIdxInvis,YnewInvis,XnewInvis],axis=1) # [U,4]
		scatterShape = tf.constant([opt.batchSize,opt.novelN,opt.H*opt.upscale,opt.W*opt.upscale,1])
		scatterML = tf.stack([MLnewInvis],axis=1) # [U,1]
		upNewML = tf.scatter_nd(scatterIdx,scatterML,scatterShape) # [B,N,uH,uW,1]
		upNewML = tf.reshape(upNewML,[opt.batchSize*opt.novelN,opt.H*opt.upscale,opt.W*opt.upscale,1]) # [BN,uH,uW,1]
		# downsample back to original size
		newML = tf.nn.avg_pool(upNewML,ksize=[1,opt.upscale,opt.upscale,1],
									   strides=[1,opt.upscale,opt.upscale,1],padding="VALID") # [BN,H,W,1]
		newMaskLogitInvis = tf.reshape(newML,[opt.batchSize,opt.novelN,opt.H,opt.W,1]) # [B,N,H,W,1]
		# combine visible/invisible
		newMaskLogit = tf.where(newMaskLogitVis>0,newMaskLogitVis,
					   tf.where(newMaskLogitInvis<0,newMaskLogitInvis,tf.ones_like(newInvDepth)*(-offsetMaskLogit)))
		newDepth = 1/(newInvDepth+1e-8)-offsetDepth
	return newDepth,newMaskLogit,collision # [B,N,H,W,1]

def quaternionToRotMatrix(q):
	with tf.name_scope("quaternionToRotMatrix"):
		qa,qb,qc,qd = tf.unstack(q,axis=1)
		R = tf.transpose(tf.stack([[1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],
								   [2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],
								   [2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)]]),perm=[2,0,1])
	return R

def transParamsToHomMatrix(q,t):
	with tf.name_scope("transParamsToHomMatrix"):
		N = tf.shape(q)[0]
		R = quaternionToRotMatrix(q)
		Rt = tf.concat([R,tf.expand_dims(t,-1)],axis=2)
		hom_aug = tf.concat([tf.zeros([N,1,3]),tf.ones([N,1,1])],axis=2)
		RtHom = tf.concat([Rt,hom_aug],axis=1)
	return RtHom

def get3DhomCoord(XYZ,opt):
	with tf.name_scope("get3DhomCoord"):
		ones = tf.ones([opt.batchSize,opt.outViewN,opt.outH,opt.outW])
		XYZhom = tf.transpose(tf.reshape(tf.concat([XYZ,ones],axis=1),[opt.batchSize,4,opt.outViewN,-1]),perm=[0,2,1,3])
	return XYZhom # [B,V,4,HW]

def get3DhomCoord2(XYZ,opt):
	with tf.name_scope("get3DhomCoord"):
		ones = tf.ones([opt.batchSize,1,opt.outViewN*opt.outH*opt.outW])
		XYZhom = tf.concat([XYZ,ones],axis=1)
	return XYZhom # [B,4,VHW]
