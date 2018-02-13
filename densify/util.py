import numpy as np

def parseObj(fname):
	vertex,edge,face = [],[],[]
	# parse vertices
	with open(fname) as file:
		for line in file:
			token = line.strip().split(" ")
			if token[0]=="v":
				vertex.append([float(token[1]),float(token[2]),float(token[3])])
	vertex = np.array(vertex)
	# parse faces
	with open(fname) as file:
		for line in file:
			token = line.strip().split()
			if len(token)>0 and token[0]=="f":
				idx1 = int(token[1].split("/")[0])-1
				idx2 = int(token[2].split("/")[0])-1
				idx3 = int(token[3].split("/")[0])-1
				# check if good triangle
				M = vertex[[idx1,idx2,idx3]]
				if np.linalg.matrix_rank(M)==3:
					face.append([idx1,idx2,idx3])
	face = np.array(face)
	# parse edges
	for f in face:
		edge.append([min(f[0],f[1]),max(f[0],f[1])])
		edge.append([min(f[0],f[2]),max(f[0],f[2])])
		edge.append([min(f[1],f[2]),max(f[1],f[2])])
	edge = [list(s) for s in set([tuple(e) for e in edge])]
	edge = np.array(edge)
	return vertex,edge,face

def removeWeirdDuplicate(F):
	F.sort(axis=1)
	F = [f for f in F]
	F.sort(key=lambda x:[x[0],x[1],x[2]])
	N = len(F)
	for i in range(N-1,-1,-1):
		if F[i][0]==F[i-1][0] and F[i][1]==F[i-1][1] and F[i][2]==F[i-1][2]:
			F.pop(i)
	return F

def edgeLength(V,E,i):
	return np.linalg.norm(V[E[i][0]]-V[E[i][1]])

def pushEtoFandFtoE(EtoF,FtoE,E,f,v1,v2):
	if v1>v2: v1,v2 = v2,v1
	e = np.where(np.all(E==[v1,v2],axis=1))[0][0]
	EtoF[e].append(f)
	FtoE[f].append(e)

def pushAndSort(Elist,V,E,ei):
	l = edgeLength(V,E,ei)
	if edgeLength(V,E,ei)>edgeLength(V,E,Elist[0]):
		Elist.insert(0,ei)
	else:
		left,right = 0,len(Elist)
		while left+1<right:
			mid = (left+right)//2
			if edgeLength(V,E,ei)>edgeLength(V,E,Elist[mid]):
				right = mid
			else:
				left = mid
		Elist.insert(left+1,ei)

def densify(V,E,F,EtoF,FtoE,Elist):
	vi_new = len(V)
	ei_new = len(E)
	# longest edge
	eL = Elist.pop(0)
	# create new vertex
	vi1,vi2 = E[eL][0],E[eL][1]
	v_new = (V[vi1]+V[vi2])/2
	V.append(v_new)
	# create new edges
	e_new1 = np.array([vi1,vi_new])
	e_new2 = np.array([vi2,vi_new])
	E.append(e_new1)
	E.append(e_new2)
	EtoF.append([])
	EtoF.append([])
	# push Elist and sort
	pushAndSort(Elist,V,E,ei_new)
	pushAndSort(Elist,V,E,ei_new+1)
	# create new triangles
	for f in EtoF[eL]:
		fi_new = len(F)
		vio = [i for i in F[f] if i not in E[eL]][0]
		f_new1 = np.array([(vi_new if i==vi2 else i) for i in F[f]])
		f_new2 = np.array([(vi_new if i==vi1 else i) for i in F[f]])
		F.append(f_new1)
		F.append(f_new2)
		e_new = np.array([vio,vi_new])
		E.append(e_new)
		EtoF.append([])
		e_out1 = [e for e in FtoE[f] if min(E[e][0],E[e][1])==min(vi1,vio) and
										max(E[e][0],E[e][1])==max(vi1,vio)][0]
		e_out2 = [e for e in FtoE[f] if min(E[e][0],E[e][1])==min(vi2,vio) and
										max(E[e][0],E[e][1])==max(vi2,vio)][0]
		# update EtoF and FtoE
		EtoF[e_out1] = [(fi_new if fi==f else fi) for fi in EtoF[e_out1]]
		EtoF[e_out2] = [(fi_new+1 if fi==f else fi) for fi in EtoF[e_out2]]
		EtoF[ei_new].append(fi_new)
		EtoF[ei_new+1].append(fi_new+1)
		EtoF[-1] = [fi_new,fi_new+1]
		FtoE.append([(e_out1 if i==e_out1 else ei_new if i==eL else len(EtoF)-1) for i in FtoE[f]])
		FtoE.append([(e_out2 if i==e_out2 else ei_new+1 if i==eL else len(EtoF)-1) for i in FtoE[f]])
		FtoE[f] = []
		pushAndSort(Elist,V,E,len(EtoF)-1)
	# # # delete old edge
	E[eL] = np.ones_like(E[eL])*np.nan
	EtoF[eL] = []
	# delete old triangles
	for f in EtoF[eL]:
		F[f] = np.ones_like(F[f])*np.nan
