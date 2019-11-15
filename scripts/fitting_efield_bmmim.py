#!/usr/bin/env python3
# ver 0.1 - coding python by Hyuntae Jung on 10/12/2018

import numpy as np
import scipy
from scipy.optimize import curve_fit, minimize, LinearConstraint, Bounds, least_squares
from scipy import stats

_angstrom_to_bohr = 1.8897259886

def fit_partial_charges(atoms,xyz_atoms,xyz_probe,e_xyz,mode='fit',trial_partial_charges=None):
	"""
	fitting partial charges with electric field

	Parameters
	----------
	atoms : np.shape(n_atoms), character
		atomic types for monomers
	xyz_atoms : np.shape(n_atoms,3), dtype=np.float
		coordinate of atoms of monomers (in unit of bohr)
		We assume coordinates are the same at all different dimer configurations
			and also the ordering is the same as variable atoms.
	xyz_probe : np.shape(n_config,3), dtype=np.float
		coordinate of probe atom at n configurations (in unit of bohr)
		We assume the last line in reading file is the coordinate for probe atom
	e_xyz : np.shape(n_config,3), dtype=np.float
		electric field for probe atom at n configurations (in unit of atomic unit)
		We assume the the ordering is the same as probe atom
	mode : 'fit' or 'trial'
		For fitting, choose 'fit'. For checking matching, choose 'trial'
	trial_partial_charges : np.array(None,2), dtype=character
		initial trial for fitting when you choose 'fit' mode. Or test the trial when you choose 'trial'
		first column: atom type, second column: partial charges

	Returns
	---------
	unique_atom_types : list of (unique) atom types
	res.x : optimized partial charges for atom types
	"""
	if ('fit' not in mode) and ('trial' not in mode):
		raise ValueError("wrong mode {}".format(mode))

	# run
	if 'fit' in mode:
		# fitting magnitude of electric field 
		res = minimize(efield_func_min, partial_charges, method='Nelder-Mead', jac=efield_der,
			args=(xyz_atoms,xyz_probe,e_xyz,unique_indices), options={'disp': True})
		# fitting component of electric field
		#res = minimize(efield_func_comp_min, partial_charges, method='Nelder-Mead', jac=efield_der,
		#	args=(xyz_atoms,xyz_probe,e_xyz,unique_indices,0), options={'disp': True}) # x component
		#res = minimize(efield_func_comp_min, res.x, method='Nelder-Mead', jac=efield_der,
		#	args=(xyz_atoms,xyz_probe,e_xyz,unique_indices,1), options={'disp': True}) # y component
		#res = minimize(efield_func_comp_min, res.x, method='Nelder-Mead', jac=efield_der,
		#	args=(xyz_atoms,xyz_probe,e_xyz,unique_indices,2), options={'disp': True}) # z component
		return unique_atom_types, res.x
	if 'trial' in mode:
		trial_ef = efield_func(partial_charges,xyz_atoms,xyz_probe,unique_indices)
		# test magnitude of electric field
		trial_ef_mag = np.sqrt(np.einsum('...i,...i',trial_ef,trial_ef))
		ef = np.sqrt(np.einsum('...i,...i',e_xyz,e_xyz))
		slope, intercept, r_value, p_value, std_err = stats.linregress(ef,trial_ef_mag)
		# test component of electric field
		slope, intercept, r_value1, p_value, std_err = stats.linregress(e_xyz[:,0],trial_ef[:,0])
		slope, intercept, r_value2, p_value, std_err = stats.linregress(e_xyz[:,1],trial_ef[:,1])
		slope, intercept, r_value3, p_value, std_err = stats.linregress(e_xyz[:,2],trial_ef[:,2])
		out = np.array([r_value, r_value1, r_value2, r_value3])
		return out, np.column_stack((trial_ef_mag, trial_ef))

def efield_func(qs,xyz_atoms,xyz_probe,unique_indices):
	"""
	electric field calculation
	"""
	trial_ef = np.zeros_like(xyz_probe)
	for i_probe in range(len(xyz_probe)):
		r = xyz_atoms - xyz_probe[i_probe]
		r2 = np.einsum('...i,...i',r,r) # dot product
		denom = 1.0/r2 # denominator for electric field
		unit_r = r/np.sqrt(r2[:,None]) # unit vector for r
		# difference between electric field from calculation and quantum result
		out = np.einsum('i,i,ij->j',qs,denom,unit_r)
		trial_ef[i_probe] = out
	return trial_ef

def efield_func_min(qs,xyz_atoms,xyz_probe,ef,unique_indices):
	"""
	fitting magnitude for electric field
	"""
	del_ef = 0.0
	for i_probe in range(len(ef)):
		r = xyz_atoms - xyz_probe[i_probe]
		r2 = np.einsum('...i,...i',r,r) # dot product
		denom = 1.0/r2 # denominator for electric field
		unit_r = r/np.sqrt(r2[:,None]) # unit vector for r
		# difference between electric field from calculation and quantum result
		out = np.einsum('i,i,ij->j',qs,denom,unit_r) - ef[i_probe]
		del_ef += np.dot(out,out)

	return del_ef

def efield_func_min_comp(qs,xyz_atoms,xyz_probe,ef,unique_indices,axis,frame):
	"""
	fitting magnitude for electric field
	"""
	del_ef = 0.0
	i_probe = frame
	r = xyz_atoms - xyz_probe[i_probe]
	r2 = np.einsum('...i,...i',r,r) # dot product
	denom = 1.0/r2 # denominator for electric field
	unit_r = r/np.sqrt(r2[:,None]) # unit vector for r
	# difference between electric field from calculation and quantum result
	out = np.einsum('i,i,ij->j',qs,denom,unit_r) - ef[i_probe]
	#del_ef += out[axis]*out[axis]
	if out[axis] < 0.0:
		del_ef = np.abs(out[axis])*10000
	else:
		del_ef = out[axis]

	return del_ef

def efield_der(qs,xyz_atoms,xyz_probe,ef,unique_indices):
	"""
	derivative of electic field calculation with repect to q_i
	"""
	der = np.zeros_like(qs)
	for i_probe in range(len(ef)):
		r = xyz_atoms - xyz_probe[i_probe]
		r2 = np.einsum('...i,...i',r,r) # dot product
		denom = 1.0/r2 # denominator for electric field
		unit_r = r/np.sqrt(r2[:,None]) # unit vector for 
		#org_fun = np.einsum('i,i,ij->j',q,denom,unit_r) - ef[i_probe]
		full_der = np.einsum('i,ij->ij',denom,unit_r) 
		for i in range(len(der)):
			list_qi = np.where(unique_indices == unique_indices[i])
			out_vec = np.sum(full_der[list_qi],axis=0)
			der[i] += np.einsum('...i,...i',out_vec,out_vec)
	return der

def efield_der_comp(qs,xyz_atoms,xyz_probe,ef,unique_indices,axis,frame):
	"""
	derivative of electic field calculation with repect to q_i
	"""
	der = np.zeros_like(qs)
	i_probe = frame
	r = xyz_atoms - xyz_probe[i_probe]
	r2 = np.einsum('...i,...i',r,r) # dot product
	denom = 1.0/r2 # denominator for electric field
	unit_r = r/np.sqrt(r2[:,None]) # unit vector for 
	#org_fun = np.einsum('i,i,ij->j',q,denom,unit_r) - ef[i_probe]
	full_der = np.einsum('i,ij->ij',denom,unit_r) 
	for i in range(len(der)):
		list_qi = np.where(unique_indices == unique_indices[i])
		out_vec = np.sum(full_der[list_qi],axis=0)
		der[i] += out_vec[axis]*out_vec[axis]
	return der


def read_efield(filename):
	""" 
	read efield file for cation - H atom configurations

	Parameters
	---------
	filename : characters
		it contains E_xyz for probe atoms 

	Returns
	-------
	E_xyz : np.shape(n_config,3), dtype=np.float
		electric field for probe atom at n configurations (in unit of atomic unit)
		We assume the the ordering is the same as probe atom
	is_none_efield : np.shape(none,1), dtype=np.int
		indice list where efield value does not exist
	"""
	E_xyz = []
	is_none_efield = []
	with open(filename,'r') as f:
		line = f.readline()
		iline = 0
		while line:
			data = line.split()
			if len(data) == 0: # E_xyz is not shown in the file 
				is_none_efield.append(iline)
				line = f.readline()
				iline += 1
			else:
				E_xyz.append(data)
				line = f.readline()
				iline += 1

	E_xyz = np.array(E_xyz,dtype=np.float32)
	is_none_efield = np.array(is_none_efield,dtype=np.int)
	return E_xyz, is_none_efield

def read_probe(filename):
	""" 
	read xyz file for monomer

	Parameters
	---------
	filename : characters
		it contains xyz coordinate of atoms for probe atoms

	Returns
	-------
	xyz_probe : np.shape(n_config,3), dtype=np.float
		coordinate of probe atom at n configurations (in unit of bohr)
		We assume the last line in reading file is the coordinate for probe atom
	"""
	f = open(filename,'r')
	lines = f.readlines()
	n_probes = int(lines[0])
	data = np.array([line.split() for line in lines[2:]])
	data = data[:,1:4]
	xyz_probe = np.array(data,dtype=np.float32)*_angstrom_to_bohr # convert unit: angstrom -> bohr
	if n_probes != len(xyz_probe):
		raise ValueError("your probe files does not match with n_probles")
	return xyz_probe

def read_monomer(filename):
	"""
	read xyz file for monomer

	Parameters
	----------
	filename : characters
		it contains xyz coordinate of atoms for monomer

	Returns
	-------
	atoms : np.shape(n_atoms), character
		atomic types for monomers
	xyz_atoms : np.shape(n_atoms,3), dtype=np.float
		coordinate of atoms of monomers (in unit of bohr)
		We assume coordinates are the same at all different dimer configurations
	
	Note
	----
	the original coordinate is in unit of angstrom, so need to convert to aotmic unit
		to compute electric field easily.
	"""
	f = open(filename,'r')
	lines = f.readlines()
	data = [line.split() for line in lines[2:]]
	n_atoms = len(data) 
	np_data = np.array(data)
	xyz_atoms = np.array(np_data[:,1:4],np.float32)*_angstrom_to_bohr # convert unit: angstrom -> bohr
	atoms = np_data[:,0]
	return atoms, xyz_atoms

def gen_linear_constraints(atoms,total_charge):
	"""
	generate linear constraints for minimize in scipy
	https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html#scipy.optimize.LinearConstraint

	Parameters
	----------
	atoms: list of character
		list of atom types for all atoms of single monomer, 
	total_charge : float
		total charge of the monomer

	Returns
	-------
	scipy.LinearConstraint object

	"""
	# get unique_atom_types
	atoms = np.array(atoms)
	n_atoms = len(atoms)
	unique_atom_types, unique_indices = np.unique(atoms, return_inverse=True)
	#print("atoms: ",atoms)
	#print("unique atom types: ",unique_atom_types)
	#print("unique indices: ",unique_indices)
	# make constraints for duplicates
	constraints_matrix=[]
	constraints_bounds=[]
	for i in range(n_atoms):
		temp_list = np.where(unique_indices == unique_indices[i])[0]
		n_duplicates = len(temp_list)
		#print(i)
		if n_duplicates > 1: # if duplicates of atom types exist, make constraints for duplicates.
			if temp_list[0] < i: # if we already looked duplicates, pass
				continue 
			pairs = []
			for k1 in range(n_duplicates):
				for k2 in range(k1+1,n_duplicates):
					#print(temp_list[k1],temp_list[k2])
					pairs.append([temp_list[k1],temp_list[k2]])
			for pair in pairs:
				base = np.zeros(n_atoms)
				base[pair[0]] = 1.0
				base[pair[1]] = -1.0
				#print(base)
				constraints_matrix.append(base)
				constraints_bounds.append([0.,0.]) # lower and upper bound
	# make constrains for total charge of monomer
	constraints_matrix.append((np.empty(n_atoms)).fill(1))
	constraints_bounds.append([total_charge,total_charge])
	constraints_matrix = np.array(constraints_matrix)
	constraints_bounds = np.array(constraints_bounds)
	#print("constraints_matrix :",constraints_matrix)
	#print("constraints_bounds :",constraints_bounds)
	out = LinearConstraint(constraints_matrix,constraints_bounds[:,0],constraints_bounds[:,1])
	return out, unique_indices
	
def fit_minimize():
	'''
	not ready for public. still on construction
	'''

	## construct linear constraints
	linear, unique_indices = gen_linear_constraints(atom_types, args.charge)

	# manually
	eq_cons = {'type':'eq',
			   'fun' : lambda x: np.array([x[1]-x[4],x[2]-x[3],x[6]-x[7],np.sum(x)-args.charge]),
			   'jac' : lambda x: np.array([[0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			   							   [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			   							   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
			   							   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])}

	# run fitting electric field with constraint for atom type
	lower_bounds = -5.0
	upper_bounds = 5.0
	ftol=1e-6
	partial_charges = popt[unique_indices]
	bounds = Bounds(np.full(n_atoms,lower_bounds),np.full(n_atoms,upper_bounds))
	res = minimize(efield_func_min, partial_charges, method='SLSQP', jac=efield_der,
			args=(xyz_atoms,xyz_probe,e_xyz,unique_indices), constraints=eq_cons, bounds=bounds, options={'ftol': ftol,'disp': True})
	
	print(res.x)
	#res = minimize(efield_func_min_comp, partial_charges, method='SLSQP', jac=efield_der_comp,
	#		args=(xyz_atoms,xyz_probe,e_xyz,unique_indices,0,0), constraints=eq_cons, bounds=bounds, options={'ftol': ftol,'disp': True}) # x component
	#print(res.x)
	# loop cycle to get better result. but not necessary for this case.
	ftol=1e-9
	#i_loop = 0
	#while i_loop < 1:
	#	print(" current ftol = {}".format(ftol))
	#	res = minimize(efield_func_min, partial_charges, method='SLSQP', jac=efield_der,
	#			args=(xyz_atoms,xyz_probe,e_xyz,unique_indices), constraints=eq_cons, bounds=bounds, options={'ftol': ftol,'disp': True})
	#	#res = minimize(efield_func_min_comp, partial_charges, method='SLSQP', jac=efield_der_comp,
	#	#		args=(xyz_atoms,xyz_probe,e_xyz,unique_indices,0,0), constraints=eq_cons, bounds=bounds, options={'ftol': ftol,'disp': True}) # x component
	#	if res.success:
	#		print("difference = {}".format(np.abs(np.max(partial_charges - res.x))))
	#		#if np.abs(np.max(partial_charges - res.x)) < 0.0001:
	#		#	print("converged less than 0.0001")
	#		#	break
	#		partial_charges = res.x
	#		ftol = ftol/1.1
	#	else:
	#		ftol = ftol*1.5
	#	i_loop += 1

	#res = least_squares(efield_func_min, partial_charges, jac=efield_der, bounds=(-5.0,5), args=(xyz_atoms,xyz_probe,e_xyz,unique_indices), verbose=1)
	res = minimize(efield_func_min, partial_charges, method='COBYLA', 
		args=(xyz_atoms,xyz_probe,e_xyz,unique_indices), constraints=linear, bounds=bounds, options={'disp': True})
	print("result= ",res.x)
	return

if __name__=='__main__':
	'''
	how to use
	----------
	python ../../../github_upload/sapt_ff_tools/fitting_efield.py -i_mon bmmim_ua.xyz -i_prob probe.xyz -i_e efield.inp -mod fit-test -i_fit partial_charges.fit

	'''
	import argparse
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
		description='partial charge fitting of monomer for united atom model with electric field')
	## args
	parser.add_argument('-i_mon', '--input_monomer', default='ua.xyz', nargs='?', 
		help='input xyz file for UA model (instead of atomic symbol, use atomic type for constraints)')
	parser.add_argument('-i_prob', '--input_probe', default='probe.xyz', nargs='?', 
		help='xyz coordinates for probe atom at all different configurations')
	parser.add_argument('-i_e', '--input_efield', default='efield.inp', nargs='?', 
		help='electric fields (Ex, Ey, Ez) for the probe atom')
	parser.add_argument('-mode', '--mode', default='fit', nargs='?', 
		help='choose mode: fitting electric field? or checking trial partial charge? (fit/test/fit-test)')
	parser.add_argument('-i_test', '--i_test', default='None', nargs='?', 
		help='input test partial charges (first column: atom type, second column: partial charge) or None')
	parser.add_argument('-charge', '--charge', default=1.0, nargs='?', type=float,
		help='total charge for the momonmer for constraint in minimization')
	parser.add_argument('-o', '--output', default='partial_charges', nargs='?', 
		help='output prefix for partial charge')
	parser.add_argument('args', nargs=argparse.REMAINDER)
	parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
	## read args
	args = parser.parse_args()
	## Check arguments for log
	print(" input arguments: {0}".format(args))

	## construct input arrays
	# read files
	atom_types, xyz_atoms = read_monomer(args.input_monomer)
	n_atoms = len(atom_types)
	xyz_probe = read_probe(args.input_probe)
	e_xyz, is_none_efield = read_efield(args.input_efield)
	# delete xyz_probe where efield does not exist
	xyz_probe = np.delete(xyz_probe, is_none_efield, axis=0)
	print(" reduced data size into {} by {}".format(len(xyz_probe),len(is_none_efield)))
	
	# get unique_atom_types
	atom_types = np.array(atom_types)
	n_atoms = len(atom_types)
	unique_atom_types, unique_indices = np.unique(atom_types, return_inverse=True)
	print(unique_atom_types,unique_indices)

	# function to fit charges
	def electric_field_curve_fit(x,*args):
		temp_atom_type = np.array(args)
		trial_ef = np.empty(len(x))
		for i_probe in x:
			r = xyz_atoms - xyz_probe[i_probe]
			r2 = np.einsum('...i,...i',r,r) # dot product
			denom = 1.0/r2 # denominator for electric field
			unit_r = r/np.sqrt(r2[:,None]) # unit vector for r
			#temp_atom_type = np.array([a,b,c,d,e,f,g,h])
			qs = temp_atom_type[unique_indices] 
			out = np.einsum('i,i,ij->j',qs,denom,unit_r)
			penalty = (np.sum(qs) - 1.0)*100
			trial_ef[i_probe] = np.sqrt(np.dot(out,out)) + penalty
		return trial_ef

	# function to fit charges
	def electric_field_curve_fit_const(x,a,b,g,h):
		c=0.48860
		d=0.46080
		e=-0.00530
		f=0.07280
		trial_ef = np.empty(len(x))
		for i_probe in x:
			r = xyz_atoms - xyz_probe[i_probe]
			r2 = np.einsum('...i,...i',r,r) # dot product
			denom = 1.0/r2 # denominator for electric field
			unit_r = r/np.sqrt(r2[:,None]) # unit vector for r
			temp_atom_type = np.array([a,b,c,d,e,f,g,h])
			qs = temp_atom_type[unique_indices] 
			out = np.einsum('i,i,ij->j',qs,denom,unit_r)
			penalty = (np.sum(qs) - 1.0)*100
			trial_ef[i_probe] = np.sqrt(np.dot(out,out)) + penalty
		return trial_ef

	# use curve_fit
	#  I tried to do scipy.optimize.minimize, but it fails to get good value.
	if 'fit' in args.mode:
		xdata = np.arange(len(xyz_probe)) # probe id
		ydata = np.sqrt(np.einsum('...i,...i',e_xyz,e_xyz)) # magnitude of electric field	
		#popt, pcov = curve_fit(electric_field_curve_fit,xdata,ydata,bounds=(-5.0,5.0),p0=np.zeros(n_atoms))
		popt, pcov = curve_fit(electric_field_curve_fit_const,xdata,ydata,bounds=(-5.0,5.0))
		# save minimization results
		#result = np.column_stack([atom_types,popt[unique_indices]])
		new_qs = np.array([popt[0],popt[1],0.48860,0.46080,-0.00530,0.07280,popt[2],popt[3]])
		result = np.column_stack([atom_types,new_qs[unique_indices]])
		print(" optimized partial charges (will be saved in {}):".format(args.output+'.fit'))
		print(result)
		np.savetxt(args.output+'.fit', result, delimiter=' ', fmt='%s', comments='')
	
	# make trial partial_charges or read
	if 'test' in args.mode:
		if 'None' not in args.i_test:
			f=open(args.i_test,'r')
			lines=f.readlines()
			data = [line.split() for line in lines]
			data = np.array(data)
			trial_data = np.array(data[:,1],dtype=np.float32)
			if len(trial_data) == n_atoms:
				partial_charges = trial_data
				print(" Done with reading charge parameters")
			else:
				raise ValueError("not correct input size for args.i_test ({} != {})"
					.format(n_atoms,np.shape(trial_data)))
		elif 'fit' in args.mode:
			#partial_charges = popt[unique_indices]
			partial_charges = new_qs[unique_indices]
		else:
			raise RuntimeError("no input charges to test")

		# test results with trial charges
		trial_ef = efield_func(partial_charges,xyz_atoms,xyz_probe,unique_indices) # electric field
		trial_ef_mag = np.sqrt(np.einsum('...i,...i',trial_ef,trial_ef)) # magnitude of electric field
		ef = np.sqrt(np.einsum('...i,...i',e_xyz,e_xyz)) # reference electric field
		slope, intercept, r_value, p_value, std_err = stats.linregress(ef,trial_ef_mag)
		# test component of electric field
		slope, intercept, r_value1, p_value, std_err = stats.linregress(-1.0*e_xyz[:,0],trial_ef[:,0])
		slope, intercept, r_value2, p_value, std_err = stats.linregress(-1.0*e_xyz[:,1],trial_ef[:,1])
		slope, intercept, r_value3, p_value, std_err = stats.linregress(-1.0*e_xyz[:,2],trial_ef[:,2])
		r_values = np.array([r_value, r_value1, r_value2, r_value3])
		trial_ef= np.column_stack((trial_ef_mag, trial_ef))
		# check linear relationship for partial_charges we got
		print(" linregress results: r_values for [E_mag, E_xyz] = {}".format(r_values))
		ef = np.sqrt(np.einsum('...i,...i',e_xyz,e_xyz))
		ef_out = np.column_stack((ef,-1.0*e_xyz)) # sign = electric field direction
		np.savetxt(args.output+'.linear', np.column_stack([ef_out,trial_ef]), 
			header='abs(e) for QM[1] and MM[5], and e_xyz for QM[2:4] and MM[6:8]')

	print(" Done: charge fitting/test for electric field")