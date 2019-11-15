# generate input file for fitting partial charges of UA model
#  for fitting_efield.py
# requires: xyz file and molpro out file
# $1: # xyz files
# $2: prefix for xyz files and out files

n_xyz=$1
prefix=$2

log='.out'
xyz='.xyz'

# after copy xyz file from generate_grid
i=0
echo $n_xyz > probe.xyz
echo "xyz probe atom" >> probe.xyz
while [ $i -lt $n_xyz ];do
	tail -1 $prefix$i$xyz >> probe.xyz # add probe_atom 
	let i=$i+1	
done

# collect electric fields for all configurations of bmmim - H
if [ -f efield.inp ]; then
	rm efield.inp
fi

i=0
while [ $i -lt $n_xyz ]; do
	out=$(grep "TOTAL EF" $prefix$i$log | awk '{print $3 " " $4 " " $5}')
	echo $out >> efield.inp
	let i=$i+1
done

# output file = probe.xyz, efield.inp
