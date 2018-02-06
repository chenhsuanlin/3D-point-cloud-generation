for r in {1..25}; do
	python3 train.py --group=0 --model=orig-ft --arch=original --load=orig-pre_it500000 --lr=1e-5 --fromIt=$((($r-1)*4000)) --toIt=$(($r*4000));
	# python3 train.py --group=0 --model=resnet-ft --arch=resnet --load=resnet-pre_it500000 --lr=1e-5 --fromIt=$((($r-1)*4000)) --toIt=$(($r*4000));
done
