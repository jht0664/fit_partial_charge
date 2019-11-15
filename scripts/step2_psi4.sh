
filename=$1
templ=$2
charge=$3
multi=$4
xc_value=$5
ext='.xyz'


declare -a fold=('sig0.8' 'sig0.9' 'sig1.0' 'sig1.4')

for ifd in "${fold[@]}"; do
	i=0
	python ../scripts/xyz_to_psi4_efield.py $ifd"/"$filename$i$ext $templ $ifd"/"$ifd".dat"
	cd $ifd
	sed -i "s/CHARGE/${charge}/g" *.dat
	sed -i "s/SHIFT_VAL/${xc_value}/g" *.dat
	sed -i "s/MULTI/${multi}/g" *.dat
	if [ -f grid.dat ]; then
		rm grid.dat
	fi
	while [ $i -lt 1000 ]; do
		tail -1 $filename$i$ext >> grid.dat
		let i=$i+1
	done
	sed -i 's/H//g' grid.dat
	cd ..
done

