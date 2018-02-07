import numpy as np
import bpy

def setupBlender(buffer_path,RESOLUTION):
	scene = bpy.context.scene
	camera = bpy.data.objects["Camera"]
	camera.data.type = "ORTHO"
	camera.data.ortho_scale = 1
	# compositor nodes
	scene.render.use_antialiasing = False
	scene.render.alpha_mode = "TRANSPARENT"
	scene.render.image_settings.color_depth = "16"
	scene.render.image_settings.color_mode = "RGBA"
	scene.render.image_settings.use_zbuffer = True
	scene.render.use_compositing = True
	scene.use_nodes = True
	tree = scene.node_tree
	for n in tree.nodes:
		tree.nodes.remove(n)
	rl = tree.nodes.new("CompositorNodeRLayers")
	fo = tree.nodes.new("CompositorNodeOutputFile")
	fo.base_path = buffer_path
	fo.format.file_format = "OPEN_EXR"
	fo.file_slots.new("Z")
	tree.links.new(rl.outputs["Z"],fo.inputs["Z"])
	scene.render.resolution_x = RESOLUTION
	scene.render.resolution_y = RESOLUTION
	scene.render.resolution_percentage = 100
	return scene,camera,fo

def setCameraExtrinsics(camera,camPos,q):
	camera.rotation_mode = "QUATERNION"
	camera.location[0] = camPos[0]
	camera.location[1] = camPos[1]
	camera.location[2] = camPos[2]
	camera.rotation_quaternion[0] = q[0]
	camera.rotation_quaternion[1] = q[1]
	camera.rotation_quaternion[2] = q[2]
	camera.rotation_quaternion[3] = q[3]
	camera.data.sensor_height = camera.data.sensor_width

def projectionMatrix(scene,camera):
	scale = camera.data.ortho_scale
	scale_u,scale_v = scene.render.resolution_x/scale,scene.render.resolution_y/scale
	u_0 = scale_u/2.0
	v_0 = scale_v/2.0
	skew = 0 # only use rectangular pixels
	P = np.array([[scale_u,      0,u_0],
				  [0      ,scale_v,v_0]])
	return P

def cameraExtrinsicMatrix(q,camPos):
	R_world2bcam = quaternionToRotMatrix(q).T
	t_world2bcam = -1*R_world2bcam.dot(np.expand_dims(np.array(camPos),-1))
	R_bcam2cv = np.array([[ 1, 0, 0],
						  [ 0,-1, 0],
						  [ 0, 0,-1]])
	R_world2cv = R_bcam2cv.dot(R_world2bcam)
	t_world2cv = R_bcam2cv.dot(t_world2bcam)
	Rt = np.concatenate([R_world2cv,t_world2cv],axis=1)
	q_world2bcam = rotMatrixToQuaternion(R_world2bcam)
	return q_world2bcam,t_world2bcam

def objectCenteredCamPos(rho,azim,elev):
	phi = np.deg2rad(elev)
	theta = np.deg2rad(azim)
	x = rho*np.cos(theta)*np.cos(phi)
	y = rho*np.sin(theta)*np.cos(phi)
	z = rho*np.sin(phi)
	return [x,y,z]

def camPosToQuaternion(camPos):
	[cx,cy,cz] = camPos
	q1 = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
	camDist = np.linalg.norm([cx,cy,cz])
	cx,cy,cz = cx/camDist,cy/camDist,cz/camDist
	t = np.linalg.norm([cx,cy])
	tx,ty = cx/t,cy/t
	yaw = np.arccos(ty) 
	yaw = 2*np.pi-np.arccos(ty) if tx>0 else yaw
	pitch = 0
	roll = np.arccos(np.clip(tx*cx+ty*cy,-1,1))
	roll = -roll if cz<0 else roll
	q2 = quaternionFromYawPitchRoll(yaw,pitch,roll)	
	q3 = quaternionProduct(q2,q1)
	return q3

def camRotQuaternion(camPos,theta):
	theta = np.deg2rad(theta)
	[cx,cy,cz] = camPos
	camDist = np.linalg.norm([cx,cy,cz])
	cx,cy,cz = -cx/camDist,-cy/camDist,-cz/camDist
	qa = np.cos(theta/2.0)
	qb = -cx*np.sin(theta/2.0)
	qc = -cy*np.sin(theta/2.0)
	qd = -cz*np.sin(theta/2.0)
	return [qa,qb,qc,qd]

def quaternionProduct(q1,q2): 
	qa = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	qb = q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
	qc = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	qd = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return [qa,qb,qc,qd]

def quaternionFromYawPitchRoll(yaw,pitch,roll):
	c1 = np.cos(yaw/2.0)
	c2 = np.cos(pitch/2.0)
	c3 = np.cos(roll/2.0)
	s1 = np.sin(yaw/2.0)
	s2 = np.sin(pitch/2.0)
	s3 = np.sin(roll/2.0)
	qa = c1*c2*c3+s1*s2*s3
	qb = c1*c2*s3-s1*s2*c3
	qc = c1*s2*c3+s1*c2*s3
	qd = s1*c2*c3-c1*s2*s3
	return [qa,qb,qc,qd]

def quaternionToRotMatrix(q):
	R = np.array([[1-2*(q[2]**2+q[3]**2),2*(q[1]*q[2]-q[0]*q[3]),2*(q[0]*q[2]+q[1]*q[3])],
				  [2*(q[1]*q[2]+q[0]*q[3]),1-2*(q[1]**2+q[3]**2),2*(q[2]*q[3]-q[0]*q[1])],
				  [2*(q[1]*q[3]-q[0]*q[2]),2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2)]])
	return R

def rotMatrixToQuaternion(R):
	t = R[0,0]+R[1,1]+R[2,2]
	r = np.sqrt(1+t)
	qa = 0.5*r
	qb = np.sign(R[2,1]-R[1,2])*np.abs(0.5*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]))
	qc = np.sign(R[0,2]-R[2,0])*np.abs(0.5*np.sqrt(1-R[0,0]+R[1,1]-R[2,2]))
	qd = np.sign(R[1,0]-R[0,1])*np.abs(0.5*np.sqrt(1-R[0,0]-R[1,1]+R[2,2]))
	return [qa,qb,qc,qd]

def randomRotation():
	pos = np.inf
	while np.linalg.norm(pos)>1:
		pos = np.random.rand(3)*2-1
	pos /= np.linalg.norm(pos)
	phi = np.arcsin(pos[2])
	theta = np.arccos(pos[0]/np.cos(phi))
	if pos[1]<0: theta = 2*np.pi-theta
	elev = np.rad2deg(phi)
	azim = np.rad2deg(theta)
	rho = 1
	theta = np.random.rand()*360
	return rho,azim,elev,theta

def getFixedViews(FIXED):
	if FIXED==4:
		camPosAll = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=float)
		camPosAll /= np.sqrt(3)
	elif FIXED==6:
		camPosAll = np.array([[1,0,0],[0,-1/np.sqrt(2),1/np.sqrt(2)],[0,1/np.sqrt(2),-1/np.sqrt(2)],
							  [-1,0,0],[0,1/np.sqrt(2),1/np.sqrt(2)],[0,-1/np.sqrt(2),-1/np.sqrt(2)]],dtype=float)
	elif FIXED==8:
		camPosAll = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[1,1,-1],[1,-1,1],[-1,1,1],[-1,-1,-1]],dtype=float)
		camPosAll /= np.sqrt(3)
	elif FIXED==12:
		camPosAll = np.array([
			[-0.6000,0,-0.8000],
			[0.4472,0,-0.8944],
			[-0.0472,0.8507,-0.5236],
			[-0.8472,0.5257,0.0764],
			[-0.8472,-0.5257,0.0764],
			[-0.0472,-0.8507,-0.5236],
			[0.8472,0.5257,-0.0764],
			[0.0472,0.8507,0.5236],
			[-0.4472,0.0000,0.8944],
			[0.0472,-0.8507,0.5236],
			[0.8472,-0.5257,-0.0764],
			[0.6000,0,0.8000]],dtype=float)
	elif FIXED==20:
		camPosAll = np.array([
			[-0.9614,0.2750,0],
			[-0.5332,0.8460,0],
			[-0.8083,-0.1155,-0.5774],
			[-0.8083,-0.1155,0.5774],
			[-0.1155,0.8083,-0.5774],
			[-0.1155,0.8083,0.5774],
			[-0.2855,0.2141,-0.9342],
			[-0.2855,0.2141,0.9342],
			[-0.5605,-0.7473,-0.3568],
			[-0.5605,-0.7473,0.3568],
			[0.5605,0.7473,-0.3568],
			[0.5605,0.7473,0.3568],
			[0.2855,-0.2141,-0.9342],
			[0.2855,-0.2141,0.9342],
			[0.1155,-0.8083,-0.5774],
			[0.1155,-0.8083,0.5774],
			[0.8083,0.1155,-0.5774],
			[0.8083,0.1155,0.5774],
			[0.5332,-0.8460,0],
			[0.9614,-0.2750,0]],dtype=float)
	else: camPosAll = None
	return camPosAll
