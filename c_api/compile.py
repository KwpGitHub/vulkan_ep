import os
import re
import sys

dir = './shaders'
cmd_remove = ''
null_out = ''
if sys.platform.find('win32') != -1:
	cmd_remove = 'del'
	null_out = ' >>nul 2>nul'
elif sys.platform.find('linux') != -1:
	cmd_remove = 'rm'
	null_out = ' > /dev/null 2>&1'
headfile = open('./include/shaders/spv_shader.hpp', 'w')
headfile.write('namespace kernel{ namespace shaders {\n')
list = os.listdir(dir)
for i in range(0, len(list)):
	if (os.path.splitext(list[i])[-1] != '.comp'):
		continue
	prefix = os.path.splitext(list[i])[0];
	path = os.path.join(dir, list[i])

	bin_file = prefix + '.tmp'
	cmd = ' glslangValidator -V ' + path + ' -S comp -o ' + bin_file
	print('compiling')
	if os.system(cmd) != 0:
		continue

	size = os.path.getsize(bin_file)

	spv_txt_file = prefix + '.spv'
	cmd = 'glslangValidator -V ' + path + ' -S comp -o ' + spv_txt_file + ' -x' + null_out
	os.system(cmd)
	infile_name = spv_txt_file
	outfile_name = './include/shaders/' + prefix + '_spv.cpp'
	array_name = prefix + '_spv'
	infile = open(infile_name, 'r')
	outfile = open(outfile_name, 'w')
	outfile.write('#include <cstdlib>\n namespace kernel { namespace shaders {\n\n')
	fmt = 'extern const unsigned int %s[%d] = {\n' % (array_name, size/4)
	outfile.write(fmt)
	for eachLine in infile:
		if(re.match(r'^.*\/\/', eachLine)):
			continue
		newline = '    ' + eachLine.replace('\t','')
		outfile.write(newline)
	infile.close()
	outfile.write("};\n")
	outfile.write("}} //namespace kernel, shaders")
	fmt = '\textern const unsigned int %s[%d];\n' % (array_name, size/4)

	headfile.write(fmt)


	os.system(cmd_remove + ' ' + bin_file)
	os.system(cmd_remove + ' ' + spv_txt_file)
headfile.write("}}")
headfile.close()