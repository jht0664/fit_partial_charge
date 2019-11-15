# user variable
prefix=$1 # output prefix
input=$2 # optimized molecule file name

declare -a dis=('0.8' '0.9' '1.0' '1.4')

for idi in "${dis[@]}"; do
	python ../scripts/grid_efield.py -i $input -s $idi
	i=0;while [ $i -lt 1000 ]; do mv "monomer_h_"$i".xyz" $prefix$i".xyz"; let i=$i+1; done
	mv monomer_h_ref.xyz $prefix"_ref.xyz"
	newf="sig"$prefix
	mkdir $newf
	mv *.xyz $newf"/"
done

