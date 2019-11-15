
import sys
xyz_file=sys.argv[1]
temp1=sys.argv[2]
output_file=sys.argv[3]

input1 = open(xyz_file, 'r')
lines1 = input1.readlines()
input1.close()
n_atoms=int(lines1[0])-1

template = open(temp1, 'r')
lines2 = template.readlines()

output = open(output_file,'w')

output.writelines(lines2[0:4]) 
output.writelines(lines1[2:n_atoms+2]) # all coordinates except probe
output.writelines(lines2[4:])

