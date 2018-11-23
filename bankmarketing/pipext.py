import os
import sys
import re
import subprocess
import argparse
def main():
	parser = argparse.ArgumentParser(description='Pip extension for vendorization')
	parser.add_argument('command',choices=['install','download'],help='install or download')
	parser.add_argument('-r',metavar='filename', nargs='?',default='requirements.txt',help='the requirements file path')
	parser.add_argument('-d',metavar='vendor',nargs='?',default='vendor',help='the vendor folder name')
	parser.add_argument('--platform',metavar='platform' ,nargs='?',help='target platform for package install or download')
	parser.add_argument('--only-binary',metavar='binary-only package names',nargs='?',help='the binary package names separated by comma')
	parser.add_argument('--only-source',metavar='source-only package names',nargs='?',help='the source package names separated by comma')
	args = parser.parse_args()
	
	#read all entries in requirements file
	with open(args.r) as f:
		source_packages = [x for x in (line.strip() for line in f) if x!='' and not x.startswith('#')]
		
	#extract all binary packages with versions
	binary_packages = list()
	binary_names = list()
	if args.only_binary:
		binary_names = re.split('[,]+',args.only_binary)
		binary_packages = [x for x in source_packages if re.split('[>=]',x)[0] in binary_names]
		source_packages = [x for x in source_packages if x not in binary_packages]
	
	#extract all source package names
	source_names = list()
	if args.only_source:
		source_names = re.split('[,]+',args.only_source)
		
	#run command to install/download binary packages if any
	print('Perform {} of {} binary packages...'.format(args.command,len(binary_packages)))
	
	for p in binary_packages:
		try:
			params = list([sys.executable,'-m','pip',args.command])
			if args.platform:
				params.extend(['--platform',args.platform])
			params.append('--only-binary=:all:')
			if args.command == 'download':
				params.extend(['-d',args.d])
			params.append(p)
			subprocess.check_call(params)
		except subprocess.CalledProcessError:
			print('{} error for {}'.format(args.command,p))
	
	#run command to install/download non-binary packages
	print('Perform {} of {} source packages...'.format(args.command,len(source_packages)))
	for p in source_packages:
		try:
			params = list([sys.executable,'-m','pip',args.command])
			params.append('--no-binary=:all:')
			if args.command == 'download':
				params.extend(['-d',args.d])
			params.append(p)
			subprocess.check_call(params)
		except subprocess.CalledProcessError:
			print('{} error for {}'.format(args.command,p))
	if args.command == 'download':
		vendor_folder = str(args.d)
		
		print('Remove conflict file format and rename wheel files in vendor folder..')
		for filename in os.listdir(vendor_folder):
			if _is_source_extension(filename):
				parts = filename.split('-')
				
				if parts[0] in binary_names:
					print('Remove file {} as it must be in binary format'.format(filename))
					os.remove(os.path.join(vendor_folder,filename))
					continue
			if _is_binary_extension(filename):
				parts = filename.split('-')
				
				if parts[0] in source_names:
					print('Remove file {} as it must be in source format'.format(filename))
					os.remove(os.path.join(vendor_folder,filename))
					continue
				if len(parts) > 2:
					new_name = parts[0]+'-'+parts[1]+'-py2.py3-none-any.whl'
					if filename!= new_name:
						print('Renaming {} to {}'.format(filename,new_name))
						if os.path.isfile(os.path.join(vendor_folder,new_name)):
							print('Remove file {} as the new file already exists'.format(filename))
							os.remove(os.path.join(vendor_folder,filename))
						else:
							os.rename(os.path.join(vendor_folder,filename),os.path.join(vendor_folder,new_name))
def _is_source_extension(filename):
	return filename.endswith('.zip') or filename.endswith('.tar.gz') or filename.endswith('.tar.bz2') or filename.endswith('.tar.Z') or filename.endswith('.tar')

def _is_binary_extension(filename):
	return filename.endswith('.whl')

if __name__ == '__main__':
	main()
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
	
	
	
	
	
	
	
	
	
	
	
	


		