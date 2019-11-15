
import sys
xyz_file=sys.argv[1]
temp1=sys.argv[2]

input1 = open(xyz_file, 'r')
lines1 = input1.readlines()
input1.close()
n_atoms=int(lines1[0])

template = open(temp1, 'r')
lines2 = template.readlines()

newfile = xyz_file.replace('.xyz','.com')
output = open(newfile,'w')

output.writelines(lines2[0:16]) 
output.writelines(lines1[2:n_atoms+2]) # all coordinates
output.writelines(lines2[16:])
output.writelines("EF,{}".format(n_atoms))


