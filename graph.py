import numpy as np
import tensorflow as tf
import time
import scipy.ndimage.filters

# build encoder
def encoder(opt,image): # [B,H,W,3]
	def conv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim])
		conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def linearLayer(opt,feat,outDim,final=False):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu if not final else fc
	with tf.variable_scope("encoder"):
		feat = image
		with tf.variable_scope("conv1"): feat = conv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("conv2"): feat = conv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("conv3"): feat = conv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("conv4"): feat = conv2Layer(opt,feat,256) # 4x4
		feat = tf.reshape(feat,[opt.batchSize,-1])
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,1024)
		with tf.variable_scope("fc3"): feat = linearLayer(opt,feat,512,final=True)
		latent = feat
	return latent

# build decoder
def decoder(opt,latent):
	def linearLayer(opt,feat,outDim):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu
	def deconv2Layer(opt,feat,outDim):
		H,W = int(feat.shape[1]),int(feat.shape[2])
		resize = tf.image.resize_images(feat,[H*2,W*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim],stddev=opt.std)
		conv = tf.nn.conv2d(resize,weight,strides=[1,1,1,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def pixelconv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[1,1,int(feat.shape[-1]),outDim],gridInit=True)
		conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
		return conv
	with tf.variable_scope("decoder"):
		feat = tf.nn.relu(latent)
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feat,1024)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])
		with tf.variable_scope("deconv1"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv"): feat = pixelconv2Layer(opt,feat,opt.outViewN*4) # 128x128
		XYZ,maskLogit = tf.split(feat,[opt.outViewN*3,opt.outViewN],axis=3) # [B,H,W,3V],[B,H,W,V]
	return XYZ,maskLogit # [B,H,W,3V],[B,H,W,V]

# build encoder
def encoder_resnet(opt,image): # [B,H,W,3]
	def conv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim])
		conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def linearLayer(opt,feat,outDim,final=False):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu if not final else fc
	with tf.variable_scope("encoder"):
		feat = image
		with tf.variable_scope("conv1"): feat = conv2Layer(opt,feat,32) # 32x32
		with tf.variable_scope("conv2"): feat = conv2Layer(opt,feat,64) # 16x16
		with tf.variable_scope("conv3"): feat = conv2Layer(opt,feat,128) # 8x8
		with tf.variable_scope("conv4"): feat = conv2Layer(opt,feat,256) # 4x4
		feat = tf.reshape(feat,[opt.batchSize,-1])
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feat,1024)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,256,final=True)
		latent = feat
	return latent

# build decoder
def decoder_resnet(opt,latent):
	def linearLayer(opt,feat,outDim):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu
	def deconv2Layer(opt,feat,outDim):
		H,W = int(feat.shape[1]),int(feat.shape[2])
		resize = tf.image.resize_images(feat,[H*2,W*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim],stddev=opt.std)
		conv = tf.nn.conv2d(resize,weight,strides=[1,1,1,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def residualBlock(opt,feat):
		dim = int(feat.shape[-1])
		feat_identity = feat
		with tf.variable_scope("conva"):
			weight,bias = createVariable(opt,[3,3,dim,dim],stddev=opt.std)
			conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
			batchnorm = batchNormalization(opt,conv,type="conv")
			feat = tf.nn.relu(batchnorm)
		with tf.variable_scope("convb"):
			weight,bias = createVariable(opt,[3,3,dim,dim],stddev=opt.std)
			conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
			batchnorm = batchNormalization(opt,conv,type="conv")
			feat = tf.nn.relu(batchnorm)
		feat += feat_identity
		return feat
	def pixelconv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[1,1,int(feat.shape[-1]),outDim],gridInit=True)
		conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
		return conv
	with tf.variable_scope("decoder"):
		feat = tf.nn.relu(latent)
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feat,1024)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,4096)
		with tf.variable_scope("fc3"): feat = linearLayer(opt,feat,16384)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])
		with tf.variable_scope("resblock1"): feat = residualBlock(opt,feat)
		with tf.variable_scope("deconv1"): feat = deconv2Layer(opt,feat,512) # 8x8
		with tf.variable_scope("resblock2"): feat = residualBlock(opt,feat)
		with tf.variable_scope("deconv2"): feat = deconv2Layer(opt,feat,256) # 16x16
		with tf.variable_scope("resblock3"): feat = residualBlock(opt,feat)
		with tf.variable_scope("deconv3"): feat = deconv2Layer(opt,feat,128) # 32x32
		with tf.variable_scope("resblock4"): feat = residualBlock(opt,feat)
		with tf.variable_scope("deconv4"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("resblock5"): feat = residualBlock(opt,feat)
		with tf.variable_scope("deconv5"): feat = deconv2Layer(opt,feat,32) # 128x128
		with tf.variable_scope("pixelconv"): feat = pixelconv2Layer(opt,feat,opt.outViewN*4) # 128x128
		XYZ,maskLogit = tf.split(feat,[opt.outViewN*3,opt.outViewN],axis=3) # [B,H,W,3V],[B,H,W,V]
	return XYZ,maskLogit # [B,H,W,3V],[B,H,W,V]

# auxiliary function for creating weight and bias
def createVariable(opt,weightShape,biasShape=None,stddev=None,gridInit=False):
	if biasShape is None: biasShape = [weightShape[-1]]
	weight = tf.Variable(tf.random_normal(weightShape,stddev=opt.std),dtype=np.float32,name="weight")
	if gridInit:
		X,Y = np.meshgrid(range(opt.outW),range(opt.outH),indexing="xy") # [H,W]
		X,Y = X.astype(np.float32),Y.astype(np.float32)
		initTile = np.concatenate([np.tile(X,[opt.outViewN,1,1]),
								   np.tile(Y,[opt.outViewN,1,1]),
								   np.ones([opt.outViewN,opt.outH,opt.outW],dtype=np.float32)*opt.renderDepth,
								   np.zeros([opt.outViewN,opt.outH,opt.outW],dtype=np.float32)],axis=0) # [4V,H,W]
		biasInit = np.expand_dims(np.transpose(initTile,axes=[1,2,0]),axis=0) # [1,H,W,4V]
	else:
		biasInit = tf.constant(0.0,shape=biasShape)
	bias = tf.Variable(biasInit,dtype=np.float32,name="bias")
	return weight,bias

# batch normalization wrapper function
def batchNormalization(opt,input,type):
	with tf.variable_scope("batchNorm"):
		globalMean = tf.get_variable("mean",shape=[input.shape[-1]],dtype=tf.float32,trainable=False,
											initializer=tf.constant_initializer(0.0))
		globalVar = tf.get_variable("var",shape=[input.shape[-1]],dtype=tf.float32,trainable=False,
										  initializer=tf.constant_initializer(1.0))
		if opt.training:
			if type=="conv": batchMean,batchVar = tf.nn.moments(input,axes=[0,1,2])
			elif type=="fc": batchMean,batchVar = tf.nn.moments(input,axes=[0])
			trainMean = tf.assign_sub(globalMean,(1-opt.BNdecay)*(globalMean-batchMean))
			trainVar = tf.assign_sub(globalVar,(1-opt.BNdecay)*(globalVar-batchVar))
			with tf.control_dependencies([trainMean,trainVar]):
				output = tf.nn.batch_normalization(input,batchMean,batchVar,None,None,opt.BNepsilon)
		else: output = tf.nn.batch_normalization(input,globalMean,globalVar,None,None,opt.BNepsilon)
	return output

# L1 loss
def l1_loss(input):
	return tf.reduce_sum(tf.abs(input))
# L1 loss (masked)
def masked_l1_loss(diff,mask):
	return l1_loss(tf.boolean_mask(diff,mask))
# sigmoid cross-entropy loss
def cross_entropy_loss(logit,label):
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=label))
