import numpy as np
#import os
import re
from collections import OrderedDict
#import matplotlib.pyplot as plt
#Modules below this line are used for unlabeled arrays
#==========================
#import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO



'''Fortran namelist reader for EFIT/EFUND file formats '''
'''I. Stewart        Columbia University      11/12/21 '''


def namelist_read(file0, silent=True, b_arr=False):
	#Return a dictionary with the parameters in the namelist file (file0)
	#b_arr refers to an array at the bottom of the file, if one exists

	f = open(str(file0), 'r')
	f_lines = f.readlines()

	datalines = [] #Initialize new dictionary of information

	if b_arr == True:
		brk_idx = [] #Find the last break and turn it into the bottom array
		equal_idx = [] #Find the last equals sign

		for i in range(len(f_lines)):
			brk_loc = f_lines[i].find('/')  #Find the lines with backslashes (line breaks)
			if brk_loc >= 0.0:
				brk_idx.append(i)

			equal_loc = f_lines[i].find('=')
			if equal_loc >= 0.0:
				equal_idx.append(i)

		f_lines[brk_idx[np.argmin(abs(brk_idx-np.max(equal_idx)))]] = 'b_arr = '

	#Prune everything after the comments and remove line breaks 
	#for line in f_lines:
	end_idx = len(f_lines)

	for i in range(len(f_lines)):
		line = f_lines[i]

		if line.find('comment:') >= 0: #Find the comment at the end (if it exists and make this the end of the file)
			end_idx = i

		line_update = line.replace('\n','')  #Remove line breaks
		line_update = line_update.replace(',',' ')   #Remove commas
		line_update = line_update.replace('"', '')  #Remove comment/quotation marks
		comment_loc = line_update.find('!')  #Find the comment
		brk_loc = line_update.find('/')  #Find the lines with backslashes (line breaks)
		headr_loc = line_update.find('&') #and list headers (& symbols)

		if i < end_idx: #Only include lines before the end comment
			if headr_loc < 0: #Remove hardcoded linebreaks and list headers 
				if brk_loc < 0:
					if comment_loc < 0:  #If there are no comments in the line (-1)
						if line_update.strip() != '':
							datalines.append(line_update.strip())
					elif comment_loc >= 0:
						#ignore everything after the comment and get rid of white space (strip)
						line_update = line_update[:comment_loc].strip()
						if line_update != '':
							datalines.append(line_update)


	data_dict = OrderedDict()#{}
	block_idx = []

	#match = re.findall('=', datalines)
	for i in range(len(datalines)):
		equal_loc = datalines[i].find('=')  #Find where the equal sign is in the array
		if equal_loc > 0:
			block_idx.append(i)
			key0 = datalines[i][:equal_loc].strip()
			data_dict[key0] = 0.0
			#print key0
			#data = datalines[i][equal_loc:].strip()
			#print i 
			#data2 = re.compile(r'(\d+)').findall(datalines[i])
			#data2 = re.compile(r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?').findall(datalines[i])
		#print datalines[i]
		#data = line.split('=')
		#print data
		#data_dict[key0.strip()] = data.strip()
	key_list = list(data_dict.keys())


	for i in range(len(key_list)):
		if key_list[i] != 'b_arr':
			#Concetenate each of the blocks (arrays that span multiple lines) of data
			if i < len(key_list)-1:
				mult_block = ' '.join(datalines[block_idx[i]:block_idx[i+1]])
				equal_loc = mult_block.find('=')
				mult_block = mult_block[equal_loc+1:].strip()
				#data_dict[key_list[i]] = np.array(mult_block.split(' '))#.astype('float')
				dupl_loc = mult_block.find('*')
				if dupl_loc >= 0:
					mult_block = mult_block.split()
					star_idx = [idx for idx, s in enumerate(mult_block) if '*' in s]

					for j in range(len(star_idx)):
						dupl_str = mult_block[star_idx[j]].split('*')
						try:
							dupl_arr = ' '.join(int(dupl_str[0])*[dupl_str[1]])
						except:
							continue
						mult_block[star_idx[j]] = dupl_arr

					mult_block = ' '.join(mult_block)

					try:
						data_dict[key_list[i]] = np.array(mult_block.split()).astype('float')

					except:
						data_dict[key_list[i]] = mult_block
				else:
					try:
						if len(mult_block.split()) > 1:  #If longer than one entry, make it an array
							data_dict[key_list[i]] = np.array(mult_block.split()).astype('float')
						else:    #if it is one entry, make it a float or an integer
							data_dict[key_list[i]] = eval(mult_block.split()[0])

					except:
						data_dict[key_list[i]] = mult_block

			#Try the same process for the last entry		
			else:
				mult_block = ' '.join(datalines[block_idx[i]:len(datalines)])  #On the last block, go to the end of the file
				equal_loc = mult_block.find('=')
				mult_block = mult_block[equal_loc+1:].strip()
				dupl_loc = mult_block.find('*')
				if dupl_loc >= 0:
					mult_block = mult_block.split()
					star_idx = [idx for idx, s in enumerate(mult_block) if '*' in s]

					for j in range(len(star_idx)):
						dupl_str = mult_block[star_idx[j]].split('*')
						try:
							dupl_arr = ' '.join(int(dupl_str[0])*[dupl_str[1]])
						except:
							continue
						mult_block[star_idx[j]] = dupl_arr

					mult_block = ' '.join(mult_block) 
					try:
						data_dict[key_list[i]] = np.array(mult_block.split()).astype('float')
					except:
						data_dict[key_list[i]] = mult_block
				else:
					try:
						data_dict[key_list[i]] = np.array(mult_block.split()).astype('float')
					except:
						data_dict[key_list[i]] = mult_block

		elif key_list[i] == 'b_arr':
			b_arr_len = len(datalines[block_idx[i]+1:len(datalines)])
			data_dict['b_arr'] = np.zeros((b_arr_len,6))

			for j in range(b_arr_len):
				try:
					b_line = np.array(re.split(r'(\s+)', datalines[block_idx[i]+1+j]))
					b_line_space_idx = []
					b_line_rmv_idx = []

					for k in range(len(b_line)):
						if b_line[k].isspace():
							b_line_space_idx.append(k)
					
					b_line_space_len = (lambda x:[len(l) for l in x])(b_line[b_line_space_idx])

					for l in range(len(b_line_space_idx)):
						if b_line_space_len[l] < 10: #== min(b_line_space_len):
							b_line_rmv_idx.append(b_line_space_idx[l])
						else:
							b_line[b_line_space_idx[l]] = 0.0

					b_line = np.delete(b_line, b_line_rmv_idx)

					#b_line = np.array(datalines[block_idx[i]+1+j].split()).astype('float')
					data_dict['b_arr'][j][0:len(b_line)] = b_line 


				except:
					data_dict['b_arr'][j][:] = np.nan

			if 'RF' not in key_list:
				try:
					data_dict['RF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),0]
					data_dict['ZF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),1]
					data_dict['WF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),2]
					data_dict['HF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),3]
					data_dict['AF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),4]
					data_dict['AF2'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),5]
					Fcoil_in_barr = True

				except:
					print('Error in determining F-coil data')

			if 'RVS' not in key_list:
				try:
					VS_idx_start = len(data_dict['b_arr'])-len(data_dict['RSISVS']) #len(data_dict['TURNFC'])
					VS_idx_end = len(data_dict['b_arr'])#len(data_dict['RSISVS'])+VS_idx_start

					data_dict['RVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,0]
					data_dict['ZVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,1]
					data_dict['WVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,2]
					data_dict['HVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,3]
					data_dict['AVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,4]
					data_dict['AVS2'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,5]

					RVS_in_barr = True

				except:
					RVS_in_barr = False
					print('Error in determining vacuum vessel data')

			if 'RE' not in key_list:
				if data_dict['IECOIL'] == 1:
					try:
						if 'RF' in data_dict.keys():
							EC_idx_start = len(data_dict['TURNFC'])
						else:
							EC_idx_start = 0	

						if RVS_in_barr == True:
							EC_idx_end = VS_idx_start
						else:
							EC_idx_end = len(data_dict['b_arr'])

						data_dict['RE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,0]
						data_dict['ZE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,1]
						data_dict['WE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,2]
						data_dict['HE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,3]
						data_dict['ECID'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,4]

					except:
						print('Error in determining E-coil data')


	# if b_arr == True:
	# 	#bttm_arr_dict = OrderedDict()
	# 	try:
	# 		for i in range(len(f_lines)):
	# 			line = f_lines[i]
	# 			#Find the bounds for the array
	# 			brk_loc = line.find('/\n')  #Find the lines with backslashes (line breaks)
	# 			comment_loc = line.find('comment')
	# 			if brk_loc >= 0:
	# 				last_brk = i  #Index of the line with the last hard break 
	# 			if comment_loc >=0:
	# 				last_cmt = i  ##Index of the line with the last word 'comment' 
	# 		bttm_arr = f_lines[last_brk+1:last_cmt]

	# 		bttm_data = []
	# 		for line in bttm_arr:
	# 			line_update = line.replace('\n','')  #Remove line breaks
	# 			line_update = line.replace(',',' ')   #Remove commas
	# 			if line_update != '':
	# 				#line_update = line_update.strip().split('    ')
	# 				#print line_update.find('')
	# 				#line_update = line_update.replace('',np.nan)  
	# 				bttm_data.append(np.array(line_update.strip().split()[0:4]).astype('float'))
	# 		#df = pd.read_csv(StringIO(bttm_arr), sep='   ', header=None)
	
	# 		try:
	# 			#FC_data = np.concatenate(bttm_data[0:len(data_dict['TURNFC'])], axis=1)
	# 			print(data_dict['TURNFC'])
	# 			FC_data = np.asarray(bttm_data[0:len(data_dict['TURNFC'])])
	# 			data_dict['RF'] = FC_data[:,0]
	# 			data_dict['ZF'] = FC_data[:,1]
	# 			data_dict['WF'] = FC_data[:,2]
	# 			data_dict['HF'] = FC_data[:,3]
	# 			#RFC = FC_data[0]
	# 			#print RFC

	# 		except:
	# 			print('Error in determining F-coil data')

	# 	except:
	# 		print('Error Finding Unlabeled Array at the Bottom of File')

	if silent == False:
		for key in data_dict:
			print(key, data_dict[key])
	 #   	print key, data_dict[key]
	f.close()

	return data_dict

#Testing
#==================================================================================
#x = namelist_read(file='k204202.00501', silent=False)
#print data_dict['PLASMA']

#plt.plot(data_dict['XMP2'], data_dict['YMP2'], marker='o', linestyle='none')
#plt.plot(data_dict['PRESSR'])
#plt.plot(data_dict['XLIM'], data_dict['YLIM'])
#plt.show()
