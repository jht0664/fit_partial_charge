# Purpose
This program is to fit partial charges of a target molecule on the electric fields generated by hydrogen atom seperated from four different distances.
For details, see following paper;
> First-Principles United Atom Force Field for the Ionic Liquid BMIM+BF4–: An Alternative to Charge Scaling, 
> Chang Yun Son, Jesse G. McDanielJ. R. Schmidt, Qiang Cui, Arun Yethiraj, J. Phys. Chem. B 2016, 120, 14, 3560-3568
> https://doi.org/10.1021/acs.jpcb.5b12371 

# Credits
Instead of utilizing the codes by Dr. Jesse G. McDaniel [Website](https://ww2.chemistry.gatech.edu/mcdaniel/jesse-g-mcdaniel),
 Hyuntae did edit and modify user-friendly python scripts for my convenience to develop SAPT-UA force field of EMMIM and BMMIM cations.
Please cite both papers for tracking contributions;
> First-Principles United Atom Force Field for the Ionic Liquid BMIM+BF4–: An Alternative to Charge Scaling, 
> Chang Yun Son, Jesse G. McDanielJ. R. Schmidt, Qiang Cui, Arun Yethiraj, J. Phys. Chem. B 2016, 120, 14, 3560-3568
> https://doi.org/10.1021/acs.jpcb.5b12371 
>
> Methylation effect of cation on phase behavior of polymer/ionic liquids mixture,
> Hyuntae Jung, Chang Yun Son, Arun Yethiraj, (in preparation)

# Tutorial
In this tutorial, I choose EMMIM cation, 1-ethyl-2,3-dimethylimidazolium, one of ionic liquids.

## Step 0: pre-requisite
The optimized geomerty of the cation is needed.
In my case, the `example/emmim.opt` file (.xyz format) was obtained by Gaussian 09 program with the input file `example/emmim_init.gjf` 
Note that the xyz file is a converted result of Guassian check file (emmim.chk) by *newzmat* a convertion tool in Gaussian. 
```
$ cd example
$ g09 emmim.gjf >& emmim.log
$ newzmat -ichk emmim.chk -oxyz emmim.opt
```

Also, you need xc-potential for DF-KS calculations in Molpro.
See Asymptotic correction for xc-potentials in [Molpro Manual](https://www.molpro.net/info/release/doc/manual/node193.html)
Note that the (xc-)shift potential equals the difference between the HOMO energy, obtained from the respective standard Kohn-Sham calculation, and the (negative) ionisation potential of the monomer  
```
$ g09 emmim_xc.gjf >& emmim_xc.log
$ ../scripts/get_xc.sh emmim_xc.log
 xc_shift = -0.42316 - ( -383.049690621 - -383.541425282 )
  If bc is available,
 xc_shift = .068574661
Done
```
unit is A.U., e.g. atomic unit. Your compute node should have `bc` command.

## Step 1: generate xyz files with single probe (hydrogen) atom
Using grid sphere points, we make .xyz files to contain coordinates of both EMMIM and hydrogen atom at 0.8, 0.9, 1.0, and 1.4 \times sigma, where sigma is the atomic radii.
```
(from Step0)
../scripts/step1.sh emmim_h_ emmim.opt
```
Then, you will see new folders; sig0.8, sig0.9, sig1.0, and sig1.4 folders with 1001 .xyz files.

## Step 2: make input files from .xyz files generated from Step 1.
### Step 2-1: For Molpro users,
you should your value for xc potential as we did in step 0.
```
../scripts/step2.sh emmim_h_ ../templates/molpro_efield_wo_midbond.input emmim.opt .068574661
```
you will get .com files

### Step 2-2: For Psi4 users, which is a free QM package,
you should your value for xc potential as we did in step 0.
```
../scripts/step2_psi4.sh emmim_h_ ../templates/psi4_efield.input 1 1 .068574661
```
you will get .dat files as input files

## Step 3: run package
### Step 3-1: Run Molpro in NERSC Cori cluster.
Here is an example of sig0.8 folder to run Molpro with our .com files
```
#!/bin/bash -l
#SBATCH -J sig1.4
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH --array=0-999
#SBATCH -t 36:00:00
#SBATCH -o molpro.o%j
module load molpro
id=$(bc <<< ${SLURM_ARRAY_TASK_ID})
echo "current id?" $id
molpro -n 6 emmim_h_${SLURM_ARRAY_TASK_ID}.com
```
You should do all folders. After calculations, you will get .out files.
For next step, transfer the .out files to the directory with .xyz files. 

### Step 3-2: Run Psi4 in Phoenix cluster at UW-Madison Chemistry.
Here is an example of sig0.8 folder to run Psi4 with our .dat files:
```
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=12
#PBS -N sig1.4
#PBS -l walltime=200:00:00
source ~/.bashrc
conda activate psi4_env
cd /path/folder/
export PSI_SCRATCH=$PWD
psi4 -i sig1.4.dat -n 12 >& log
exit 0
```
you will get `grid_field.dat` file for E_x, E_y, and E_z.

## Step 4: manipulate input file for fitting electric fields
### Step 4-1: From molpro output files,
Again, it should run in directory where xyz files and molpro .out output files exist:
```
(in sig folders)
../scripts/fitting_efield_prepare.sh 1000 emmim_h_
```
Then, you will get `probe.xyz` and `efield.inp` in each sig folder for next step.

### 4-2: From Psi4 output files,
```
mkdir example/merge
cd example/merge
cat ../sig*/grid_field.dat >> efield.inp
echo "4000" >> probe.xyz
echo "probe atoms" >> probe.xyz
awk 'BEGIN{OFS="\t"}$1="H " $1' ../sig*/grid.dat >> probe.xyz
```
simply done for next step.

### Step 5: run `fitting_efiled.py` to get optimized partial charges
input files such as `emmim_ua.xyz`, `efield.inp`, `probe.xyz` are required.
For our SAPT-UA force field, it needs another xyz file excluding hydrogen atoms from `emmim.opt` (xyz-format) file, that is, `emmim_ua.xyz`.
Also, the xyz file should use atom type for each atom, instead of element symbol.

You should modify `scripts/fitting_efield.py` to keep some partial charges constant.
Unfortunately, I skipped to make such an user-friendly program for this.
Please take a look at it carefully, and modify to fit your purpose.
; See line 430 - 433, 440, and 456 in `scripts/fitting_efield_emmim.py`
; Those lines are necessary to change for your own target molecule.

For example, I use constraints for C3, C4, and C5 in `emmim_ua.xyz` kept constant as same as EMMIM cation SAPT-UA.
; see line 431-433
```
cd example/merge/
python ../../scripts/fitting_efield_emmim.py -i_mon ../emmim_ua.xyz -i_prob probe.xyz -i_e efield.inp -mod fit-test -charge 1.0 
```
Then, output file `partial_charges.fit` and `partial_charges.linear` are obtained.
```
# Then, if need to optimize again with previous fitting results,
python ../../scripts/fitting_efield_emmim.py -i_mon emmim_ua.xyz -i_prob probe.xyz -i_e efield.inp -mod fit-test -i_test partial_charges.fit
```

## Step 6: Result
The EMMIM cation's result looks like:
```
 input arguments: Namespace(args=[], charge=1.0, i_test='partial_charges.fit', input_efield='efield.inp', input_monomer='emmim_ua.xyz', input_probe='probe.xyz', mode='fit-test', output='partial_charges')
 reduced data size into 4000 by 0
['C1' 'C2' 'C3' 'C4' 'C5' 'C7' 'N'] [0 6 1 1 6 3 4 2 5]
 optimized partial charges (will be saved in partial_charges.fit):
[['C1' '0.3270866762291163']
 ['N' '-0.6771527680193455']
 ['C2' '0.3630559205278955']
 ['C2' '0.3630559205278955']
 ['N' '-0.6771527680193455']
 ['C4' '0.4585']
 ['C5' '0.0734']
 ['C3' '0.5165']
 ['C7' '0.25272004797989983']]
 Done with reading charge parameters
 linregress results: r_values for [E_mag, E_xyz] = [0.78352039 0.97404385 0.97987775 0.98173413]
 Done: charge fitting/test for electric field
```
R value is not good for magnitude, but x,y,z axis field looks okay.

To plot,
```
gnuplot
plot 'partial_charges.linear' u 1:4 w p, x # for magnitude of electric field
plot 'partial_charges.linear' u 2:6 w p, '' u 3:7 w p, '' u 4:8 w p, x # for x, y, z component of electric field
```

# Tips
Setting constraints on some atom types is critical because optimizer does not give the best answer with arbitrary initials.
Because there is no single set of variables which satisfies convergence criteria, unrealistic values can be chosen as a result of optimiztion.
Thus, you may need to reduce variables by using more number of constraints, and rationalize the signs and magnitude of atom types even though it was the best optimized by machine.
