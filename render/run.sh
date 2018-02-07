blender blank.blend -b -P render_fixed.py -- /dataset/ShapeNetCore.v2 $1 $1.list 128 $2
blender blank.blend -b -P render.py -- /dataset/ShapeNetCore.v2 $1 $1.list 128
python2 convertEXR.py $1 $1.list 128 $2

# blender blank.blend -b -P render_fixed.py -- /dataset/ShapeNetCore.v2 03001627 03001627.list 128 8
# blender blank.blend -b -P render.py -- /dataset/ShapeNetCore.v2 03001627 03001627.list 128
# python2 convertEXR.py 03001627 03001627.list 128 8

# ./run.sh 03001627 8
