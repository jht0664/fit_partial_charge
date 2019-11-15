log_file=$1 # log file name for xc_shift

# get epsilon for HOMO electron
if [ -f homo.log ]; then
	echo "homo.log overwrite? STOP"
	exit 0
fi
if [ -f ip.log ]; then
	echo "ip.log overwrite? STOP"
	exit 0
fi

grep "Alpha  occ. eigenvalues" $log_file > homo.log
nlines=$(wc homo.log | awk '{print $1}')
nres=$(( nlines / 2 ))
eps_homo=$( sed "${nres}q;d" homo.log | awk '{print $NF}' )
rm homo.log

# get Ionization potential
org_E=$( grep "SCF Done:  E(" $log_file | sed "1q;d" | awk '{print $5}' )
ion_E=$( grep "SCF Done:  E(" $log_file | sed "2q;d" | awk '{print $5}' )

# show values or calculation 
echo " xc_shift = "$eps_homo" - ( "$ion_E" - "$org_E" )"
echo "  If bc is available,"
ip_res=$( bc <<< "$org_E - $ion_E" )
xc_res=$( bc <<< "$eps_homo - $ip_res" )
echo " xc_shift = "$xc_res

echo "Done"
