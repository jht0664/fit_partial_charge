
filename=$1
templ=$2
mol=$3
xc_value=$4
ext='.xyz'

natoms=$(head -1 $mol)
let natoms=$natoms+1

declare -a fold=('sig0.8' 'sig0.9' 'sig1.0' 'sig1.4')

for ifd in "${fold[@]}"; do
	i=0
	while [ $i -lt 1000 ]; do
		python ../scripts/xyz_to_molpro_efield.py $ifd"/"$filename$i$ext $templ
		let i=$i+1
	done
	cd $ifd
	sed -i "s/n_atoms_monb/${natoms}/g" *.com
	sed -i "s/value_shift/${xc_value}/g" *.com
	cd ..
done

