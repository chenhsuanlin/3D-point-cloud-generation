import scipy.io
import numpy as np
import os,sys
import util
import time

SHAPENETPATH = sys.argv[-3]
CATEGORY = sys.argv[-2]
MODEL_LIST = sys.argv[-1]

densifyN = 100000

models = []
with open(MODEL_LIST) as file:
	for line in file:
		model = line.strip()
		models.append(model)
models.sort()
modelN = len(models)

output_path = "output/{0}".format(CATEGORY)
if not os.path.isdir(output_path):
	os.makedirs(output_path)

for m in models:
	timeStart = time.time()
	
	shape_file = "{2}/{0}/{1}/models/model_normalized.obj".format(CATEGORY,m,SHAPENETPATH)
	V,E,F = util.parseObj(shape_file)
	F = util.removeWeirdDuplicate(F)
	Vorig,Eorig,Forig = V.copy(),E.copy(),F.copy()

	# sort by length (maintain a priority queue)
	Elist = list(range(len(E)))
	Elist.sort(key=lambda i:util.edgeLength(V,E,i),reverse=True)

	# create edge-to-triangle and triangle-to-edge lists
	EtoF = [[] for j in range(len(E))]
	FtoE = [[] for j in range(len(F))]
	for f in range(len(F)):
		v = F[f]
		util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[0],v[1])
		util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[0],v[2])
		util.pushEtoFandFtoE(EtoF,FtoE,E,f,v[1],v[2])
	V,E,F = list(V),list(E),list(F)

	# repeat densification
	for z in range(densifyN):
		util.densify(V,E,F,EtoF,FtoE,Elist)

	densifyV = np.array(V[-densifyN:])

	scipy.io.savemat("{0}/{1}.mat".format(output_path,m),{
		"V": Vorig,
		"E": Eorig,
		"F": Forig,
		"Vd": densifyV
	})

	print("{0} done, time = {1:.6f} sec".format(m,time.time()-timeStart))
