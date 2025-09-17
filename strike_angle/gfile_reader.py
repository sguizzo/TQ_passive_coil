import numpy as np
#import os
import re
from collections import OrderedDict
#import matplotlib.pyplot as plt
from namelist_reader import namelist_read
from scipy import integrate 
#import pandas as pd
#import sys


'''    G-file reader based off of the OMFIT modules  '''
'''I. Stewart        Columbia University    11/12/21 '''
#Original can be found in omfit_gfile_reader.py
#class omfit_classes.omfit_eqdsk.OMFITgeqdsk(*args, **kwargs)[source]
#Info can be found in G_EQDSK.pdf (G EQDSK FORMAT, Lao 04/04/02)

# based on w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
def splitter(inv, step=16):
	value = []
	for k in range(len(inv) // step):
		value.append(inv[step * k : step * (k + 1)])
	return value

def merge(inv):
    if not len(inv):
        return ''
    if len(inv[0]) > 80:
        # SOLPS gEQDSK files add spaces between numbers
        # and positive numbers are preceeded by a +
        return (''.join(inv)).replace(' ', '')
    else:
        return ''.join(inv)


def gfile_read(file0, silent=True):
	#Read the input g-file (file0), and return a dictionary with the equilibrium parameters

	f = open(str(file0), 'r')
	EQDSK_lines = f.read()#.splitlines()    #Read the file
	EQDSK_lines = re.sub('\s$', '', EQDSK_lines, flags=re.MULTILINE)  #Remove any blank lines (add to OMFIT)
	#Replace 'negative' zeros which shift the text block (add to OMFIT)
	EQDSK_lines = EQDSK_lines.replace(' -0.000000000E+00', ' 0.000000000E+00')
	EQDSK_lines = EQDSK_lines.splitlines()  #Split the lines up individually

	#Initialize new dictionary of information
	data_dict = OrderedDict()

	#The first line includes description and sizes
	data_dict['CASE'] = np.array(splitter(EQDSK_lines[0][0:48], 8))

	#Find the number of grid points in the R and Z direction (NW and NH) using the first line
	try:
		tmp = list([_f for _f in EQDSK_lines[0][48:].split(' ') if _f])
		[IDUM, data_dict['NW'], data_dict['NH']] = list(map(int, tmp[:3]))

	except ValueError:  # Can happen if no space between numbers, such as 10231023
		IDUM = int(EQDSK_lines[0][48:52])
		data_dict['NW'] = int(EQDSK_lines[0][52:56])
		data_dict['NH'] = int(EQDSK_lines[0][56:60])
		tmp = []
        #printd('IDUM, NW, NH', IDUM, self['NW'], self['NH'], topic='OMFITgeqdsk.load')

	if len(tmp) > 3:
		data_dict['EXTRA_HEADER'] = EQDSK_lines[0][49 +\
		len(re.findall('%d +%d +%d ' % (IDUM, data_dict['NW'], data_dict['NH']), EQDSK_lines[0][49:])[0]) + 2 :]
	offset = 1

	#Find the next 20 parameters organized into 5 per row

    # fmt: off
	[data_dict['RDIM'], data_dict['ZDIM'], data_dict['RCENTR'], data_dict['RLEFT'], data_dict['ZMID'],
	data_dict['RMAXIS'], data_dict['ZMAXIS'], data_dict['SIMAG'], data_dict['SIBRY'], data_dict['BCENTR'],
    data_dict['CURRENT'], data_dict['SIMAG'], XDUM, data_dict['RMAXIS'], XDUM,
    data_dict['ZMAXIS'], XDUM, data_dict['SIBRY'], XDUM, XDUM] = list(map(eval, splitter(merge(EQDSK_lines[offset:offset + 4]))))
    # fmt: on
	offset = offset + 4

	#Get the next set of data arrays in increments of nlNW (increase offset by nlNW at each step)
	nlNW = int(np.ceil(data_dict['NW']/5.0))
	data_dict['FPOL'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
	offset = offset + nlNW
	data_dict['PRES'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
	offset = offset + nlNW
	data_dict['FFPRIM'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
	offset = offset + nlNW
	data_dict['PPRIME'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
	offset = offset + nlNW

	#Find the PSIRZ values as the next array in the sequence 
	try:
        #The official gEQDSK file format saves PSIRZ as a single flat array of size rowsXcols
		nlNWNH = int(np.ceil(data_dict['NW']*data_dict['NH']/5.0))
		data_dict['PSIRZ'] = np.reshape(np.fromiter(splitter(merge(EQDSK_lines[offset : offset + nlNWNH])),\
		dtype=np.float64)[: data_dict['NH'] * data_dict['NW']], (data_dict['NH'], data_dict['NW']),)
		offset = offset + nlNWNH

	except ValueError:
		# Sometimes gEQDSK files save row by row of the PSIRZ grid (eg. FIESTA code)
		nlNWNH = data_dict['NH']*nlNW
		data_dict['PSIRZ'] = np.reshape(np.fromiter(splitter(merge(EQDSK_lines[offset : offset + nlNWNH])),\
		dtype=np.float64)[: data_dict['NH'] * data_dict['NW']], (data_dict['NH'], data_dict['NW']),)
		offset = offset + nlNWNH


	data_dict['QPSI'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
	offset = offset + nlNW

	#Find the number of plasma boundary coordinates (BBBS) and number of limiter coordinates
	data_dict['NBBBS'], data_dict['LIMITR'] = list(map(int, [_f for _f in EQDSK_lines[offset : offset + 1][0].split(' ') if _f][:2]))
	offset = offset + 1

	#Use the number of plasma boundary coordinates to find the R and Z arrays (next in the sequence)
	nlNBBBS = int(np.ceil(data_dict['NBBBS'] * 2 / 5.0))
	data_dict['RBBBS'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset\
	+ nlNBBBS]))))[0::2])[: data_dict['NBBBS']]
	data_dict['ZBBBS'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset\
	+ nlNBBBS]))))[1::2])[: data_dict['NBBBS']]
	offset = offset + max(nlNBBBS, 1)


	#nlLIMITR = int(np.ceil(data_dict['LIMITR'] * 2 / 5.0))
	#print(splitter(merge(EQDSK_lines[offset : offset + nlLIMITR])))


	#Now get the limiter coordinates based on the number of points from LIMITR
	try:
		#This try/except is to handle some gEQDSK files written by older versions of ONETWO
		nlLIMITR = int(np.ceil(data_dict['LIMITR'] * 2 / 5.0))
		data_dict['RLIM'] = np.array(list(map(float,\
		splitter(merge(EQDSK_lines[offset : offset + nlLIMITR]))))[0::2])[: data_dict['LIMITR']]
		data_dict['ZLIM'] = np.array(list(map(float,\
        splitter(merge(EQDSK_lines[offset : offset + nlLIMITR]))))[1::2])[: data_dict['LIMITR']]
		offset = offset + nlLIMITR

	except ValueError:
		#If it fails, make the limiter a rectangle around the plasma boundary
		#that does not exceed the computational domain
		data_dict['LIMITR'] = 5
		dd = data_dict['RDIM'] / 10.0
		R = np.linspace(0, data_dict['RDIM'], 2) + data_dict['RLEFT']
		Z = np.linspace(0, data_dict['ZDIM'], 2) - data_dict['ZDIM'] / 2.0 + data_dict['ZMID']
		data_dict['RLIM'] = np.array([\
		max([R[0], np.min(data_dict['RBBBS']) - dd]),\
		min([R[1], np.max(data_dict['RBBBS']) + dd]),\
		min([R[1], np.max(data_dict['RBBBS']) + dd]),\
		max([R[0], np.min(data_dict['RBBBS']) - dd]),\
		max([R[0], np.min(data_dict['RBBBS']) - dd]),])

		data_dict['ZLIM'] = np.array([\
		max([Z[0], np.min(data_dict['ZBBBS']) - dd]),\
		max([Z[0], np.min(data_dict['ZBBBS']) - dd]),\
		min([Z[1], np.max(data_dict['ZBBBS']) + dd]),\
		min([Z[1], np.max(data_dict['ZBBBS']) + dd]),\
		max([Z[0], np.min(data_dict['ZBBBS']) - dd]),])


	#Get the toroial rotation switch (KVTOR), R (in m) where the rotational pressure profile is specified (RVTOR)
	#Rotational pressure profile (PRESSW), the derivative of the rotational pressure profile (PWPRIM),
	#mass density switch (NMASS), mass density on uniform poloidal flux grid (DMION),
	#normalized toroidal flux on uniform poloidal flux grid (RHOVN)
	
	try:
		[data_dict['KVTOR'], data_dict['RVTOR'], data_dict['NMASS']] =\
		list(map(float, [_f for _f in EQDSK_lines[offset : offset + 1][0].split(' ') if _f]))
		offset = offset + 1

		if data_dict['KVTOR'] > 0:
			data_dict['PRESSW'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
			offset = offset + nlNW
			data_dict['PWPRIM'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
			offset = offset + nlNW

		if data_dict['NMASS'] > 0:
			data_dict['DMION'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
			offset = offset + nlNW

		#Add the 'fundamental' normalized radial coordinate rho_N, based on the toroidal flux
		#Normalized toroidal flux on uniform poloidal flux grid
		data_dict['RHOVN'] = np.array(list(map(float, splitter(merge(EQDSK_lines[offset : offset + nlNW])))))
		offset = offset + nlNW

	except Exception:
		pass
	'''
    #Add RHOVN if missing
	if 'RHOVN' not in data_dict or not len(data_dict['RHOVN']) or not np.sum(data_dict['RHOVN']):
		data_dict.add_rhovn()
	'''

	#Fix some gEQDSK files that do not fill PRES info (eg. EAST)
	if not np.sum(data_dict['PRES']):
		pres = integrate.cumtrapz(data_dict['PPRIME'], np.linspace(data_dict['SIMAG'],\
		data_dict['SIBRY'], len(data_dict['PPRIME'])), initial=0)
		data_dict['PRES'] = pres - pres[-1]

	#Calculate the R and Z coordinates of the equilibrium from the given dimensions 
	data_dict['R'] = np.linspace(0.0, data_dict['RDIM'], data_dict['NW'])+data_dict['RLEFT']
	data_dict['Z'] = np.linspace(0.0, data_dict['ZDIM'], data_dict['NH'])-data_dict['ZDIM']/2.0+data_dict['ZMID']

	#Include the RZ mesh grid using the calculated R and Z coordinates
	data_dict['R_grid'], data_dict['Z_grid'] = np.meshgrid(data_dict['R'], data_dict['Z'])

	#Construct R-Z mesh
    #r = np.zeros([nxefit, nyefit])
    #z = r.copy()
    #for i in range(nxefit):
    #    r[i,:] = rgrid1 + xdim*i/float(nxefit-1)
    #for j in range(nyefit):
    #    z[:,j] = (zmid-0.5*zdim) + zdim*j/float(nyefit-1)
    ###[R, Z] = np.meshgrid(aux['R'], aux['Z'])

	#Create the creates for the poloidal flux (psi "radial coordinates") and the normalized poloidal flux
	data_dict['PSI'] = np.linspace(data_dict['SIMAG'], data_dict['SIBRY'], len(data_dict['PRES']))
	data_dict['PSI_NORM'] = np.linspace(0.0, 1.0, len(data_dict['PRES']))

	#The normalized effective radius ("rho poloidal")
	data_dict['RHOp'] = np.sqrt(data_dict['PSI_NORM'])

	#Now read the "auxiliary" parameters (Listed parameters in the gfile)
	aux_dict = namelist_read(file0=file0, silent=True)

	#If the parameter from the aux_dict is not in data_dict, add it
	for key in aux_dict:
		if key not in data_dict:
			data_dict[key] = aux_dict[key] 

	 #if raw and add_aux:
        # add AuxQuantities and fluxSurfaces
    #    self.addAuxQuantities()
    #    self.addFluxSurfaces(**self.OMFITproperties)
    #elif not raw:
        # Convert tree representation to COCOS 1
     #   self._cocos = self.native_cocos()
     #   self.cocosify(1, calcAuxQuantities=True, calcFluxSurfaces=True)



	if silent == False:
		for key in data_dict:
			print(key, data_dict[key])
	 #   	print key, data_dict[key]
	f.close()

	return data_dict

#Testing
#==================================================================================
# gdict = gfile_read(file0='g000019.10000', silent=True)
# #plt.plot(x['RBBBS'], x['ZBBBS'])
# #plt.plot(x['RLIM'], x['ZLIM'])

# #        if sum(self['RHOVN']) == 0.0:
# #            usePsi = True
# if 'RHOVN' in gdict:
# 	x_psi = gdict['RHOVN']

# else:
# 	if 'RHO' in gdict:
# 		x_psi = gdict['RHO']
# 	else:
# 		#usePsi = True
# 		x_psi = np.linspace(0.0, 1.0, len(gdict['PRES']))

# # if usePsi:
# #                 xName = '$\\psi$'
# #                 x = np.linspace(0, 1, len(self['PRES']))
# #             else:
# #                 xName = '$\\rho$'
# #                 if 'RHOVN' in self and np.sum(self['RHOVN']):
# #                     x = self['RHOVN']
# #                 else:
# #                     x = self['AuxQuantities']['RHO']
# #ax.plot(x, self['PPRIME'], **kw)


# #plt.plot(x_psi, gdict['PRES'])
# plt.plot(x_psi, gdict['QPSI'])
# #plt.plot(x_psi, gdict['PPRIME'])
# #plt.plot(x_psi, gdict['FFPRIM'])
# plt.show()

#print data_dict['PLASMA']

#plt.plot(data_dict['XMP2'], data_dict['YMP2'], marker='o', linestyle='none')
#plt.plot(data_dict['PRESSR'])
#plt.plot(data_dict['XLIM'], data_dict['YLIM'])
#plt.show()