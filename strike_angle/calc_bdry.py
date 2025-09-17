#import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from collections import OrderedDict
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
#import timeit
import intersect
import contourpy
#import matplotlib._contour as _contour

'''Code to calculate the coordinates of the magentic boundary (RBBBS and ZBBBS) from a gfile'''
'''Also gives the upper and lower contours of the separatrix + x-point locations            ''' 
'''Ian Stewart                     Columbia University                              12/10/21'''

#References: Marching squares algorthm, contour.py and old _cntr.py, bound.f90 in EFIT, and gs_trace_boundary.m in Toksys 


def calc_bdry(theta_coords, R_maxis, Z_maxis, R_grid, Z_grid, Psi_arr0, Psi_bdry0, xpt2=False, show_calc=False, reducebdry=0.0):
	#Interpolate the x and y coordinates on a finer grid to increase the resolution and make it easier for scikit's find_contours
	#to find the different contours

	Psi_arr = np.copy(Psi_arr0)
	Psi_bdry = np.copy(Psi_bdry0)

	x_new = np.linspace(np.min(R_grid), np.max(R_grid), 301)
	y_new = np.linspace(np.min(Z_grid), np.max(Z_grid), 301)
	X_new, Y_new = np.meshgrid(x_new, y_new)


	maxisclosest_Ridx = np.argmin(np.abs(R_grid[1,:]-R_maxis))
	maxisclosest_Zidx = np.argmin(np.abs(Z_grid[:,1]-Z_maxis))
	Psi_maxis_est = Psi_arr[maxisclosest_Zidx, maxisclosest_Ridx]

	if Psi_maxis_est < 0.0:
		Psi_arr *= -1.0
		Psi_bdry *= -1.0
		Psi_maxis_est *= -1.0

	if Psi_bdry < 0.0:
		Psi_arr += np.abs(Psi_bdry)
		Psi_bdry += np.abs(Psi_bdry)



	Psi_arr_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr), (np.ravel(X_new), np.ravel(Y_new)), method='linear')
	Psi_arr_interp = np.reshape(Psi_arr_interp, np.shape(X_new))

	_mask, _corner_mask, nchunk = None, True, 0
	#contour_generator = _contour.QuadContourGenerator(X_new, Y_new, Psi_arr_interp, _mask, _corner_mask, nchunk)
	contour_generator = contourpy.contour_generator(X_new, Y_new, Psi_arr_interp)#, _mask, _corner_mask, nchunk)


	if reducebdry != 0.0:
		#Manually reduce Psi_bdry to find the separate curves of the plasma boundary and separatrix  
		cntr_vertices = contour_generator.create_contour(Psi_bdry*reducebdry)	

	else:
		#Without a manual reduction of the boundary, determine whether the convention for flux is negative or positive
		#Needed because some codes have psi=0 at the magnetic axis and some have psi=0 at infinity 
		if Psi_bdry >= 0.0:
			delta_psi = (Psi_maxis_est - Psi_bdry)*0.01
			#cntr_vertices = contour_generator.create_contour(np.abs(Psi_bdry*1.04))  #Initialize at a slightly higher psi curve first
			cntr_vertices = contour_generator.create_contour(np.abs(Psi_bdry+delta_psi))  #Initialize at a slightly higher psi curve first
		#elif Psi_bdry == 0.0:
		#	cntr_vertices = contour_generator.create_contour(Psi_arr[maxisclosest_Zidx, maxisclosest_Ridx]*0.01) #If Psi_bdry=0, initialize at 1% of max Psi
			#cntr_vertices = contour_generator.create_contour(np.max(np.abs(Psi_arr))*0.01) #If Psi_bdry=0, initialize at 1% of max Psi
		else:
			delta_psi = (Psi_maxis_est - Psi_bdry)*0.01
			#cntr_vertices = contour_generator.create_contour(Psi_bdry*0.98)  #Initialize at a slightly lower psi curve first
			cntr_vertices = contour_generator.create_contour(Psi_bdry-delta_psi)  #Initialize at a slightly lower psi curve first

	cntrs_dict = OrderedDict() #Initialize a dictionary to hold the coordinates of the different contours
	
	#version_cntr = sys.modules[_contour.__package__].__version__  #Get the version of matplotlib._contour


	#Matplotlib 3.5.2 (Python 3.8)
	#if float(version_cntr[0]) >= 3:
	#	for i in range(len(cntr_vertices[0])):
	#		cntrs_dict['cntr'+str(i)] = np.transpose(cntr_vertices[0][i])

	#Matplotlib 2.1.1 (Python 3.6 on PSFC servers)
	#else:
	#for i in range(len(cntr_vertices)):
	#	cntrs_dict['cntr'+str(i)] = np.transpose(cntr_vertices[i])

	#New method using contourpy.contour_generator
	for i in range(len(cntr_vertices)):
		cntrs_dict['cntr'+str(i)] = np.transpose(cntr_vertices[i])


	if show_calc == True:
		import matplotlib.pyplot as plt


		plt.figure(figsize=(5, 9))
		plt.contour(R_grid, Z_grid, Psi_arr, levels=np.array([Psi_bdry]), colors='k')
		plt.contour(R_grid, Z_grid, Psi_arr, levels=np.linspace(np.min(Psi_arr) ,np.max(Psi_arr), 41), colors='grey', linestyles='-', alpha=0.5)
		for key in cntrs_dict:
			plt.plot(cntrs_dict[key][0], cntrs_dict[key][1])

		plt.gca().set_aspect('equal')
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		plt.tight_layout()#pad=0.06)
		plt.show()


	#Organize the contours by the average Z coordinate (upper_sep, plasma boundary, and lower_sep)
	cntrs_key_list = list(cntrs_dict)
	Z_avg_lst = []  #Find the average Z coordinate (centroid) for each curve
	for key in cntrs_dict:
		Z_avg_lst.append(np.mean(cntrs_dict[key][1]))

	bdry_idx = np.argmin(np.abs(np.array(Z_avg_lst)-0.0))
	cntrs_dict['plasma_bdry'] = cntrs_dict.pop(cntrs_key_list[bdry_idx])


	if len(cntrs_dict) == 3:
		upper_idx = np.argmax(np.array(Z_avg_lst))
		cntrs_dict['upper_sep'] = cntrs_dict.pop(cntrs_key_list[upper_idx])
		lower_idx = np.argmin(np.array(Z_avg_lst))
		cntrs_dict['lower_sep'] = cntrs_dict.pop(cntrs_key_list[lower_idx])


	elif len(cntrs_dict) == 5:
		sorted_cntrs_idx = np.argsort(Z_avg_lst)
		cntrs_dict['upper_sep2'] = cntrs_dict.pop(cntrs_key_list[sorted_cntrs_idx[4]])
		cntrs_dict['upper_sep'] = cntrs_dict.pop(cntrs_key_list[sorted_cntrs_idx[3]])
		cntrs_dict['lower_sep'] = cntrs_dict.pop(cntrs_key_list[sorted_cntrs_idx[1]])
		cntrs_dict['lower_sep2'] = cntrs_dict.pop(cntrs_key_list[sorted_cntrs_idx[0]])

	else:
		pass


	#Convert the plasma boundary x and y coordinates into polar coordinates (with an origin at (R_maxis, Z_maxis) >(0,0))
	r_bdry0 = np.sqrt(((cntrs_dict['plasma_bdry'][0]-R_maxis)**2)+((cntrs_dict['plasma_bdry'][1]-Z_maxis)**2))
	theta_bdry0 = np.arctan2((cntrs_dict['plasma_bdry'][1]-Z_maxis),(cntrs_dict['plasma_bdry'][0]-R_maxis))

	#Wrap the theta and r coordinates to prevent errors in the interpolation bounds
	theta_bdry0_wrapped = np.append(theta_bdry0, theta_bdry0+2.0*np.pi)
	theta_bdry0_wrapped = np.append(theta_bdry0-2.0*np.pi, theta_bdry0_wrapped)
	r_bdry0_wrapped = np.append(r_bdry0, r_bdry0)
	r_bdry0_wrapped = np.append(r_bdry0, r_bdry0_wrapped)

	#Interpolate the boundary coordinates onto the regular theta_coords from
	#the input array (want one at every degree, not scikit's default separation)
	f_interp = interp1d(theta_bdry0_wrapped, r_bdry0_wrapped, kind='linear')#, fill_value='extrapolate',bounds_error=False)
	r_bdry0_interp = f_interp(theta_coords)

	x_interp = r_bdry0_interp*np.cos(theta_coords)
	y_interp = r_bdry0_interp*np.sin(theta_coords)

	#Calculate the 2D interpolation of the grid (first part of griddata that supplies a function to call).
	#Example function call can be found in ndgriddata.py. Using CloughTocher2DInterpolator is for cubic interpolation
	#otherwise, use LinearNDInterpolator 
	from scipy.interpolate import CloughTocher2DInterpolator
	f_2Dpsi_interp = CloughTocher2DInterpolator(list(zip(np.ravel(R_grid), np.ravel(Z_grid))),\
	np.ravel(Psi_arr))#, fill_value=None)

	# import matplotlib.pyplot as plt
	# print(R_grid[0])
	# print(Z_grid[:,0])

	# for i in range(np.shape(Z_grid)[1]):
	# 	plt.plot(X_new[0], f_2Dpsi_interp(X_new, Y_new)[i])
	# 	plt.plot(R_grid[0], Psi_arr[i], color='k', marker='o', linestyle='none', markersize=3)

	   
	# plt.show()

	#, tol=1e-06, maxiter=400, rescale=False)
	#from scipy.interpolate import RegularGridInterpolator
	#f_2Dpsi_interp = RegularGridInterpolator((R_grid[0], Z_grid[:,0]), Psi_arr.T, method='cubic',\
	#fill_value=None, bounds_error=False)
	#from scipy.interpolate import RBFInterpolator
	#f_2Dpsi_interp = RBFInterpolator((R_grid, Z_grid), Psi_arr)


	x_recalc = np.copy(x_interp)+R_maxis #Initialize the recalculated R and Z coordinates ("x" and "y")
	y_recalc = np.copy(y_interp)+Z_maxis #gives the first guess as the scikit calculated boundary

	#Now recalculate are more accurate boundary using interpolation with 0.1mm radial accuracy
	for i in range(len(theta_coords)):
			r_scan = np.linspace(r_bdry0_interp[i]-0.01, r_bdry0_interp[i]+0.01, 201) #0.1 mm scan
			x_scan = r_scan*np.cos(theta_coords[i])+R_maxis
			y_scan = r_scan*np.sin(theta_coords[i])+Z_maxis
			#scan_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr), (x_scan, y_scan), method='cubic')
			scan_interp = f_2Dpsi_interp((x_scan, y_scan))

			closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

			if closest_idx == 0: #If the closest index is the first index in the array, move the scan to smaller r
				while True:
					r_scan = r_scan - 0.01
					x_scan = r_scan*np.cos(theta_coords[i])+R_maxis
					y_scan = r_scan*np.sin(theta_coords[i])+Z_maxis
					#scan_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr), (x_scan, y_scan), method='cubic')
					scan_interp = f_2Dpsi_interp((x_scan, y_scan))
					closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))
					if closest_idx > 0: 
						x_recalc[i] = x_scan[closest_idx]
						y_recalc[i] = y_scan[closest_idx]
						break
					elif r_scan[0] <= 0.05: 
						break   
				

			elif closest_idx == len(scan_interp)-1: #If the closest index is the last index in the array, move the scan to larger r
				while True:
					r_scan = r_scan + 0.01
					x_scan = r_scan*np.cos(theta_coords[i])+R_maxis
					y_scan = r_scan*np.sin(theta_coords[i])+Z_maxis
					#scan_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr), (x_scan, y_scan), method='cubic')
					scan_interp = f_2Dpsi_interp((x_scan, y_scan))
					closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))
					if closest_idx < len(scan_interp)-1: 
						x_recalc[i] = x_scan[closest_idx]
						y_recalc[i] = y_scan[closest_idx]
						break
					elif r_scan[0] >= 2.5: 
						break   
				

			else:
				x_recalc[i] = x_scan[closest_idx]
				y_recalc[i] = y_scan[closest_idx]


	#SLOPE MUST BE DECREASING
	cntrs_dict['plasma_bdry'] = np.vstack([x_recalc, y_recalc])


	#Interpolate the other contours to get more accurate results using the same 2D interpolation
	if xpt2 != False:
		for key in cntrs_dict:
			#import matplotlib.pyplot as plt
			#plt.plot(cntrs_dict[key][0], cntrs_dict[key][1], marker='o', color='k')	
			if key != 'plasma_bdry':
				for i in range(np.shape(cntrs_dict[key])[1]):
					if i < np.shape(cntrs_dict[key])[1]-1:
						m1 = (cntrs_dict[key][1][i] - cntrs_dict[key][1][i+1])/(cntrs_dict[key][0][i] - cntrs_dict[key][0][i+1])
					else:
						m1 = (cntrs_dict[key][1][i-1] - cntrs_dict[key][1][i])/(cntrs_dict[key][0][i-1] - cntrs_dict[key][0][i])

					m2 = -1.0/m1
					b2 = cntrs_dict[key][1][i] - m2*cntrs_dict[key][0][i]
					dx_line = 0.01/np.sqrt((m2**2)+1.0) #0.01 m = 1 cm scan (hypotenuse)

					x_scan = np.linspace(cntrs_dict[key][0][i]-dx_line, cntrs_dict[key][0][i]+dx_line, 201)
					y_scan = m2*x_scan+b2

					scan_interp = f_2Dpsi_interp((x_scan, y_scan))

					closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

					if closest_idx == 0:
						while True:
							x_scan = x_scan - dx_line/2.0
							y_scan = y_scan = m2*x_scan+b2
							#y_scan = y_scan - 0.04
							scan_interp = f_2Dpsi_interp((x_scan, y_scan))
							closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

							if closest_idx > 0: 
								cntrs_dict[key][0][i] = x_scan[closest_idx]
								cntrs_dict[key][1][i] = y_scan[closest_idx]
								break
							elif x_scan[0] <= np.min(R_grid): 
								break   
				
					elif closest_idx == len(scan_interp)-1:
						while True:
							x_scan = x_scan + dx_line/2.0
							y_scan = m2*x_scan+b2
							scan_interp = f_2Dpsi_interp((x_scan, y_scan))
							#scan_deriv = np.diff(scan_interp)
							closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

							if closest_idx < len(scan_interp)-1: 
								cntrs_dict[key][0][i] = x_scan[closest_idx]
								cntrs_dict[key][1][i] = y_scan[closest_idx]
								break
							elif x_scan[0] >= np.max(R_grid): 
								break   
						#print 'Too Far'
					else:
						cntrs_dict[key][0][i] = x_scan[closest_idx]
						cntrs_dict[key][1][i] = y_scan[closest_idx]
					

	else:
		for key in cntrs_dict:
			if key != 'plasma_bdry':
				for i in range(np.shape(cntrs_dict[key])[1]):
					x_scan = cntrs_dict[key][0][i]*np.ones(201)  #Scan a range of y values for one x value
					y_scan = np.linspace(cntrs_dict[key][1][i]-0.01, cntrs_dict[key][1][i]+0.01, 201) #0.1 mm scan
					scan_interp = f_2Dpsi_interp((x_scan, y_scan))
					#scan_deriv = np.diff(scan_interp)
					closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

					if closest_idx == 0:
						while True:
							y_scan = y_scan - 0.01
							scan_interp = f_2Dpsi_interp((x_scan, y_scan))
							#scan_deriv = np.diff(scan_interp)
							closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

							if closest_idx > 0: 
								cntrs_dict[key][0][i] = x_scan[closest_idx]
								cntrs_dict[key][1][i] = y_scan[closest_idx]
								break
							elif y_scan[0] <= np.min(Z_grid): 
								break   
						#print 'Too Close'
					elif closest_idx == len(scan_interp)-1:
						while True:
							y_scan = y_scan + 0.01
							scan_interp = f_2Dpsi_interp((x_scan, y_scan))
							#scan_deriv = np.diff(scan_interp)
							closest_idx = np.nanargmin(np.abs(scan_interp-Psi_bdry))

							if closest_idx < len(scan_interp)-1: 
								cntrs_dict[key][0][i] = x_scan[closest_idx]
								cntrs_dict[key][1][i] = y_scan[closest_idx]
								break
							elif y_scan[0] >= np.max(Z_grid): 
								break   
						#print 'Too Far'
					else:
						cntrs_dict[key][0][i] = x_scan[closest_idx]
						cntrs_dict[key][1][i] = y_scan[closest_idx]


	if show_calc:
		import matplotlib.pyplot as plt

		plt.figure(figsize=(5, 9))
		second_interp2D = f_2Dpsi_interp(np.linspace(np.min(R_grid), np.max(R_grid), 301),\
		np.linspace(np.min(Z_grid), np.max(Z_grid), 301))
		plt.contour(R_grid, Z_grid, Psi_arr, levels=np.array([Psi_bdry]), colors='k')
		plt.contour(R_grid, Z_grid, Psi_arr, levels=np.linspace(np.min(Psi_arr) ,np.max(Psi_arr), 41), colors='grey', linestyles='-', alpha=0.5)
		for k in cntrs_dict:
			if k != 'plasma_bdry':
				plt.plot(cntrs_dict[k][0], cntrs_dict[k][1], color='green', marker='o',markersize=2, linewidth=1.5, alpha=0.6)

		plt.plot(cntrs_dict['plasma_bdry'][0], cntrs_dict['plasma_bdry'][1], linewidth=1.5, marker='o', markersize=2)
		try:
			plt.plot(cntrs_dict['lower_sep'][0], cntrs_dict['lower_sep'][1], linewidth=1.5)
		except:
			print('No Lower Separatrix')
		try:
			plt.plot(cntrs_dict['upper_sep'][0], cntrs_dict['upper_sep'][1], linewidth=1.5)
		except:
			print('No Upper Separatrix')

		plt.plot(x_interp+R_maxis, y_interp+Z_maxis, color='r', marker='o',markersize=2, linewidth=1.5)
		plt.plot(x_recalc, y_recalc, color='cyan', marker='o',markersize=2, linewidth=1.5)
		for i in range(len(x_interp)):
			plt.plot([R_maxis, x_interp[i]+R_maxis], [Z_maxis, y_interp[i]+Z_maxis], color='grey', alpha=0.6, linewidth=0.5)


		plt.gca().set_aspect('equal')
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		plt.tight_layout()#pad=0.06)
		plt.show()	


	#Return the dictionary of coordinate contours
	return cntrs_dict



import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # https://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficie
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)   




#Now calculate the x-points
def calc_xpoints(R_grid, Z_grid, Psi_arr0, Psi_bdry0, bdry_RZ=False, Psi_restrict=False, xpt2_dict=False, show_calc=False):

	Psi_arr = np.copy(Psi_arr0)
	Psi_bdry = np.copy(Psi_bdry0)

	xpoint_dict = OrderedDict() #Initialize a dictionary to hold the coordinates of the different contours

	#if RZ_bdry.all() != False:
	if type(bdry_RZ) != bool:
		plasma_bdryR = bdry_RZ[0]
		plasma_bdryZ = bdry_RZ[1]

		centr_closest_Ridx = np.argmin(np.abs(R_grid[1,:]-(np.max(plasma_bdryR)+np.min(plasma_bdryR))/2.0))
		centr_closest_Zidx = np.argmin(np.abs(Z_grid[:,1]-(np.max(plasma_bdryZ)+np.min(plasma_bdryZ))/2.0))

		Psi_center_est = Psi_arr[centr_closest_Zidx, centr_closest_Ridx]

		if Psi_center_est < 0.0:
			Psi_arr *= -1.0
			Psi_bdry *= -1.0

		if Psi_bdry < 0.0:
			Psi_arr += np.abs(Psi_bdry)
			Psi_bdry += np.abs(Psi_bdry)


	#Do a rough interpolation of the grid to narrow down the x-point location
	X_interp, Y_interp = np.meshgrid(np.linspace(np.min(R_grid), np.max(R_grid), 401), np.linspace(np.min(Z_grid),\
	np.max(Z_grid), 401))


	#Second method (interpolate the derivative)
	#=====================================================
	dR = abs(R_grid[0,0] - R_grid[0,1])
	dZ = abs(Z_grid[0,0] - Z_grid[1,0])

	grad_psi = np.gradient((2.0*np.pi)*Psi_arr, dR, dZ)
	B_R = -grad_psi[1]/(2.0*np.pi*R_grid)
	B_Z = grad_psi[0]/(2.0*np.pi*R_grid)
	B_totp = np.sqrt((B_R**2) + (B_Z**2)) #Total field in the poloidal plane

	Btotp_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_totp), (X_interp, Y_interp), method='linear')

	#Exclude the interior region of the plasma
	Psi_rad_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr), (X_interp, Y_interp), method='linear')


	if Psi_restrict == False:
		if type(bdry_RZ) != bool:
			if Psi_bdry >= 0.0:
				delta_psi = (Psi_center_est - Psi_bdry)*0.01
				outside_idx = np.where(Psi_rad_interp >= Psi_bdry-delta_psi)

			else:
				delta_psi = (Psi_center_est - Psi_bdry)*0.01
				outside_idx = np.where(Psi_rad_interp <= Psi_bdry-delta_psi)


		else:
			#outside_idx = np.where(Psi_rad_interp >= Psi_bdry*0.9)
			if Psi_bdry >= 0.0:
				outside_idx = np.where(Psi_rad_interp >= Psi_bdry*1.10)
			else:
				outside_idx = np.where(Psi_rad_interp >= Psi_bdry*0.90)


	else:
		if Psi_bdry >= 0.0:
			outside_idx = np.where(Psi_rad_interp >= Psi_restrict)

		else:
			outside_idx = np.where(Psi_rad_interp <= Psi_restrict)

	Btotp_interp_nan = np.copy(Btotp_interp)
	Btotp_interp_nan[outside_idx] = np.nan
	Btotp_interp[outside_idx] = np.max(Btotp_interp)


	#Find the X-points (rough calc.)
	#==========================================
	if xpt2_dict == False:
		uphalf_idx = np.where(Y_interp>=0.0)
		upX_idx = np.argmin(Btotp_interp[uphalf_idx])
		upX_coord = np.array([X_interp[uphalf_idx][upX_idx], Y_interp[uphalf_idx][upX_idx]])

		lowhalf_idx = np.where(Y_interp<0.0)
		lowX_idx = np.argmin(Btotp_interp[lowhalf_idx])
		lowX_coord = np.array([X_interp[lowhalf_idx][lowX_idx], Y_interp[lowhalf_idx][lowX_idx]])

	else:
		from matplotlib.path import Path

		try:
			lim_polygon = Path(np.transpose(np.array([xpt2_dict['XLIM'], xpt2_dict['YLIM']])))
		except:
			lim_polygon = Path(np.transpose(np.array([xpt2_dict['RLIM'], xpt2_dict['ZLIM']])))

		minima_idx = detect_local_minima(Btotp_interp_nan)
		minima_points = np.vstack((X_interp[minima_idx].flatten(), Y_interp[minima_idx].flatten())).T 
		minima_inlim = lim_polygon.contains_points(minima_points)


		upMin_indices = np.where(minima_points[minima_inlim].T[1] >= 0.0)[0]
		upMinima = minima_points[minima_inlim][upMin_indices]
		upMinima_values = Btotp_interp_nan[minima_idx].flatten()[minima_inlim][upMin_indices]

		if len(upMin_indices) > 2:
			group1_idx = []
			group2_idx = []

			dist0 = np.sqrt((upMinima.T[1]-upMinima.T[1][0])**2 + \
			(upMinima.T[0]-upMinima.T[0][0])**2)
			farthest_ptidx = np.argmax(dist0)
			halfway_discrim = dist0[farthest_ptidx]/2.0
			for i in range(len(upMin_indices)):
				if dist0[i] < halfway_discrim:
					group1_idx.append(i)
				else:
					group2_idx.append(i)

			xpt1_idx = group1_idx[np.argmin(upMinima_values[group1_idx])]
			xpt2_idx = group2_idx[np.argmin(upMinima_values[group2_idx])]

		else:
			xpt1_idx = 0
			xpt2_idx = 1


		if len(upMin_indices) == 1:
			upX_coord = upMinima.T
			upX_coord2 = np.nan


		else:
			if np.abs(upMinima[xpt1_idx][1]) < np.abs(upMinima[xpt2_idx][1]):
				upX_coord = upMinima[xpt1_idx]
				upX_coord2 = upMinima[xpt2_idx]
			else:
				upX_coord = upMinima[xpt2_idx]
				upX_coord2 = upMinima[xpt1_idx]



		lowMin_indices = np.where(minima_points[minima_inlim].T[1] < 0.0)[0]
		lowMinima = minima_points[minima_inlim][lowMin_indices]
		lowMinima_values = Btotp_interp_nan[minima_idx].flatten()[minima_inlim][lowMin_indices]

		if len(lowMin_indices) > 2:
			group1_idx = []
			group2_idx = []

			dist0 = np.sqrt((lowMinima.T[1]-lowMinima.T[1][0])**2 + \
			(lowMinima.T[0]-lowMinima.T[0][0])**2)
			farthest_ptidx = np.argmax(dist0)
			halfway_discrim = dist0[farthest_ptidx]/2.0
			for i in range(len(lowMin_indices)):
				if dist0[i] < halfway_discrim:
					group1_idx.append(i)
				else:
					group2_idx.append(i)

			xpt1_idx = group1_idx[np.argmin(lowMinima_values[group1_idx])]
			xpt2_idx = group2_idx[np.argmin(lowMinima_values[group2_idx])]

		else:
			xpt1_idx = 0
			xpt2_idx = 1


		if len(lowMin_indices) == 1:
			lowX_coord = lowMinima.T
			lowX_coord2 = 0.0
		else:
			if np.abs(lowMinima[xpt1_idx][1]) < np.abs(lowMinima[xpt2_idx][1]):
				lowX_coord = lowMinima[xpt1_idx]
				lowX_coord2 = lowMinima[xpt2_idx]
			else:
				lowX_coord = lowMinima[xpt2_idx]
				lowX_coord2 = lowMinima[xpt1_idx]


	#Now find the x-points to within 0.1mm
	#===========================================
	dresol = 0.0001 #The desired resolution of the grid (in meters)
	dX_interp = (X_interp[0,0]-X_interp[0,1])*8.0
	dY_interp = (Y_interp[0,0]-Y_interp[1,0])*4.0

	X_zoom1, Y_zoom1 = np.meshgrid(np.linspace(lowX_coord[0]-dX_interp, lowX_coord[0]+dX_interp,\
	int(round(2.0*np.abs(dX_interp/dresol)))), np.linspace(lowX_coord[1]-dY_interp, lowX_coord[1]+dY_interp,\
	int(round(2.0*np.abs(dY_interp/dresol)))))


	X_zoom2, Y_zoom2 = np.meshgrid(np.linspace(upX_coord[0]-dX_interp, upX_coord[0]+dX_interp,\
	round(2.0*np.abs(dX_interp/dresol))), np.linspace(upX_coord[1]-dY_interp, upX_coord[1]+dY_interp,\
	round(2.0*np.abs(dY_interp/dresol))))


	if xpt2_dict != False:
		X_zoom3, Y_zoom3 = np.meshgrid(np.linspace(lowX_coord2[0]-dX_interp, lowX_coord2[0]+dX_interp,\
		round(2.0*np.abs(dX_interp/dresol))), np.linspace(lowX_coord2[1]-dY_interp, lowX_coord2[1]+dY_interp,\
		round(2.0*np.abs(dY_interp/dresol))))

		X_zoom4, Y_zoom4 = np.meshgrid(np.linspace(upX_coord2[0]-dX_interp, upX_coord2[0]+dX_interp,\
		round(2.0*np.abs(dX_interp/dresol))), np.linspace(upX_coord2[1]-dY_interp, upX_coord2[1]+dY_interp,\
		round(2.0*np.abs(dY_interp/dresol))))


	f_2Dbr_interp = LinearNDInterpolator(list(zip(np.ravel(R_grid), np.ravel(Z_grid))), np.ravel(B_R))
	f_2Dbz_interp = LinearNDInterpolator(list(zip(np.ravel(R_grid), np.ravel(Z_grid))), np.ravel(B_Z))

	#Br_zoom1 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_R), (X_zoom1, Y_zoom1), method='linear')
	Br_zoom1 = f_2Dbr_interp(X_zoom1, Y_zoom1)
	Y_contr1 = np.linspace(lowX_coord[1]-dY_interp, lowX_coord[1]+dY_interp, int(round(2.0*np.abs(dY_interp/dresol))))
	X_contr0 = np.linspace(lowX_coord[0]-dX_interp, lowX_coord[0]+dX_interp, int(round(2.0*np.abs(dX_interp/dresol))))
	X_contr1 = np.zeros(len(Y_contr1))

	for i in range(len(Y_contr1)):
		X_contr1[i] = X_contr0[np.argmin(np.abs(Br_zoom1[i]))]

	#Bz_cntr1 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_Z), (X_contr1.T, Y_contr1.T), method='linear')
	Bz_cntr1 = f_2Dbz_interp(X_contr1.T, Y_contr1.T)
	lowX_idx = np.argmin(np.abs(Bz_cntr1))
	lowX_coord = np.array([X_contr1[lowX_idx], Y_contr1[lowX_idx]])
	xpoint_dict['lower_xpt'] = lowX_coord


	#Br_zoom2 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_R), (X_zoom2, Y_zoom2), method='linear')
	Br_zoom2 = f_2Dbr_interp(X_zoom2, Y_zoom2)
	Y_contr2 = np.linspace(upX_coord[1]-dY_interp, upX_coord[1]+dY_interp, round(2.0*np.abs(dY_interp/dresol)))
	X_contr0 = np.linspace(upX_coord[0]-dX_interp, upX_coord[0]+dX_interp, round(2.0*np.abs(dX_interp/dresol)))
	X_contr2 = np.zeros(len(Y_contr2))

	for i in range(len(Y_contr2)):
		X_contr2[i] = X_contr0[np.argmin(np.abs(Br_zoom2[i]))]

	#Bz_cntr2 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_Z), (X_contr2.T, Y_contr2.T), method='linear')
	Bz_cntr2 = f_2Dbz_interp(X_contr2.T, Y_contr2.T)

	upX_idx = np.argmin(np.abs(Bz_cntr2))
	upX_coord = np.array([X_contr2[upX_idx], Y_contr2[upX_idx]])
	xpoint_dict['upper_xpt'] = upX_coord

	#=================================================================================
	if xpt2_dict != False:
		Br_zoom3 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_R), (X_zoom3, Y_zoom3), method='linear')
		Y_contr3 = np.linspace(lowX_coord2[1]-dY_interp, lowX_coord2[1]+dY_interp, int(round(2.0*np.abs(dY_interp/dresol))))
		X_contr0 = np.linspace(lowX_coord2[0]-dX_interp, lowX_coord2[0]+dX_interp, int(round(2.0*np.abs(dX_interp/dresol))))
		X_contr3 = np.zeros(len(Y_contr3))

		for i in range(len(Y_contr3)):
			X_contr3[i] = X_contr0[np.argmin(np.abs(Br_zoom3[i]-0.0))]

		Bz_cntr3 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_Z), (X_contr3.T, Y_contr3.T), method='linear')

		lowX_idx2 = np.argmin(np.abs(Bz_cntr3-0.0))
		lowX_coord2 = np.array([X_contr3[lowX_idx2], Y_contr3[lowX_idx2]])
		xpoint_dict['lower_xpt2'] = lowX_coord2


		Br_zoom4 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_R), (X_zoom4, Y_zoom4), method='linear')
		Y_contr4 = np.linspace(upX_coord2[1]-dY_interp, upX_coord2[1]+dY_interp, int(round(2.0*np.abs(dY_interp/dresol))))
		X_contr0 = np.linspace(upX_coord2[0]-dX_interp, upX_coord2[0]+dX_interp, int(round(2.0*np.abs(dX_interp/dresol))))
		X_contr4 = np.zeros(len(Y_contr4))

		for i in range(len(Y_contr4)):
			X_contr4[i] = X_contr0[np.argmin(np.abs(Br_zoom4[i]-0.0))]

		Bz_cntr4 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(B_Z), (X_contr4.T, Y_contr4.T), method='linear')

		upX_idx2 = np.argmin(np.abs(Bz_cntr4-0.0))
		upX_coord2 = np.array([X_contr4[upX_idx2], Y_contr4[upX_idx2]])
		xpoint_dict['upper_xpt2'] = upX_coord2

	#====================================================================================

	if show_calc == True:
		import matplotlib.pyplot as plt
		from matplotlib import gridspec

		fig1 = plt.figure(figsize=(10,8))
		gs = gridspec.GridSpec(1, 1)#, width_ratios=(35, 1))
		ax1 = plt.subplot(gs[0,0])

		full_levels = np.linspace(np.min(Psi_arr),np.max(Psi_arr),42)
		inner_lvls = full_levels[np.where((full_levels >= Psi_bdry))[0]]
		outer_lvls  = full_levels[np.where((full_levels < Psi_bdry))[0]]

		cntr1 = ax1.contour(R_grid, Z_grid, Psi_arr, levels=inner_lvls, colors='k',linestyles='-')#\
		ax1.contour(R_grid, Z_grid, Psi_arr, levels=outer_lvls, colors='k', linestyles='--', alpha=0.3)
		ax1.contour(R_grid, Z_grid, Psi_arr, levels=np.array([Psi_bdry]), colors='k', linestyles='-', linewidths=2.0)

		ax1.contour(R_grid, Z_grid, Psi_arr, levels=np.array([Psi_bdry*0.991]), colors='k', linestyles='-', linewidths=2.0)
		ax1.contourf(X_interp, Y_interp, Btotp_interp, levels=np.linspace(np.min(Btotp_interp),\
		np.max(Btotp_interp), 41))

		ax1.plot(X_contr1, Y_contr1, color='r')

		ax1.plot(upX_coord[0], upX_coord[1], color='w', marker='o')
		ax1.plot(lowX_coord[0], lowX_coord[1], color='w', marker='o')

		ax1.plot(np.ones(len(Y_contr1))*1.4, Y_contr1, color='k')

		ax1.set_xlabel('R (m)')
		ax1.set_ylabel('Z (m)')
		ax1.minorticks_on()

		ax1.set_aspect('equal')
		fig1.patch.set_facecolor('#ffffff')
		plt.tight_layout(pad=0.06)
		plt.show()
	

	return xpoint_dict



#Now calculate the strike points
def calc_strikepts(separx_R, separx_Z, lim_R, lim_Z, Xpt_R, Xpt_Z, calc_angles=False, show_calc=False):
	
	R_strikes, Z_strikes = intersect.intersection(separx_R, separx_Z, lim_R, lim_Z)
	strikept_dict = OrderedDict() #Initialize a dictionary to hold the coordinates of the strike points
	left_pts_idx = []
	right_pts_idx = []

	for i in range(len(R_strikes)):
		if R_strikes[i] <= Xpt_R:
			left_pts_idx.append(i)
		elif R_strikes[i] > Xpt_R:
			right_pts_idx.append(i)

	dist_arr = np.sqrt((R_strikes-Xpt_R)**2 + (Z_strikes-Xpt_Z)**2)


	if len(left_pts_idx) > 1: #If more than two intersections use logic to find closest to the X-point
		clst_idx = np.argmin(dist_arr[left_pts_idx])
		strikept_dict['left_strkpt'] = np.array([R_strikes[left_pts_idx][clst_idx], Z_strikes[left_pts_idx][clst_idx]]) 
	else:
		strikept_dict['left_strkpt'] = np.array([R_strikes[left_pts_idx], Z_strikes[left_pts_idx]])

	if len(right_pts_idx) > 1: #If more than two intersections use logic to find closest to the X-point
		clst_idx = np.argmin(dist_arr[right_pts_idx])
		strikept_dict['right_strkpt'] = np.array([R_strikes[right_pts_idx][clst_idx], Z_strikes[right_pts_idx][clst_idx]]) 
	else:
		strikept_dict['right_strkpt'] = np.array([R_strikes[right_pts_idx], Z_strikes[right_pts_idx]])


	#If strike points were not found, fill the entries with np.nan
	if len(left_pts_idx) == 0:
		strikept_dict['left_strkpt'] = np.array([np.nan, np.nan])

	if len(right_pts_idx) == 0:
		strikept_dict['right_strkpt'] = np.array([np.nan, np.nan])


	if calc_angles == True:
		if np.isnan(strikept_dict['right_strkpt'][0]):
			strikept_dict['rightstrkpt_angle'] = np.nan

		else:
			dist1 = np.sqrt((strikept_dict['right_strkpt'][0] - separx_R)**2 + (strikept_dict['right_strkpt'][1] - separx_Z)**2)
			dist1b = np.sqrt((separx_R[1:] - separx_R[:-1])**2 + (separx_Z[1:] - separx_Z[:-1])**2)
			is_between1 = (dist1[1:] + dist1[:-1]) - dist1b
			between1_idx = np.argmin(np.abs(is_between1))
			#dist1_idx = sorted(range(len(dist1)), key=lambda k: dist1[k])[0:2]

			dist2 = np.sqrt((strikept_dict['right_strkpt'][0] - lim_R)**2 + (strikept_dict['right_strkpt'][1] - lim_Z)**2) 
			#dist2_idx = sorted(range(len(dist2)), key=lambda k: dist2[k])[0:2]
			dist2b = np.sqrt((lim_R[1:] - lim_R[:-1])**2 + (lim_Z[1:] - lim_Z[:-1])**2)
			is_between2 = (dist2[1:] + dist2[:-1]) - dist2b
			between2_idx = np.argmin(np.abs(is_between2))


			m1 = (separx_Z[between1_idx+1] - separx_Z[between1_idx])/(separx_R[between1_idx+1]-separx_R[between1_idx]+10**(-20))
			m2 = (lim_Z[between2_idx+1] - lim_Z[between2_idx])/(lim_R[between2_idx+1]-lim_R[between2_idx]+10**(-20))
			strikept_dict['rightstrkpt_angle'] = np.abs(np.arctan2(m1-m2, 1+m1*m2)*180.0/np.pi)


		#================================================================================================
		if np.isnan(strikept_dict['left_strkpt'][0]):
			strikept_dict['leftstrkpt_angle'] = np.nan

		else:
			dist1 = np.sqrt((strikept_dict['left_strkpt'][0] - separx_R)**2 + (strikept_dict['left_strkpt'][1] - separx_Z)**2)
			#dist1b = np.sqrt((separx_R[1:] - separx_R[:-1])**2 + (separx_Z[1:] - separx_Z[:-1])**2)
			is_between1 = (dist1[1:] + dist1[:-1]) - dist1b
			between1_idx = np.argmin(np.abs(is_between1))
			#dist1_idx = sorted(range(len(dist1)), key=lambda k: dist1[k])[0:2]

			dist2 = np.sqrt((strikept_dict['left_strkpt'][0] - lim_R)**2 + (strikept_dict['left_strkpt'][1] - lim_Z)**2) 
			#dist2_idx = sorted(range(len(dist2)), key=lambda k: dist2[k])[0:2]
			#dist2b = np.sqrt((lim_R[1:] - lim_R[:-1])**2 + (lim_Z[1:] - lim_Z[:-1])**2)
			is_between2 = (dist2[1:] + dist2[:-1]) - dist2b
			between2_idx = np.argmin(np.abs(is_between2))

			m1 = (separx_Z[between1_idx+1] - separx_Z[between1_idx])/(separx_R[between1_idx+1]-separx_R[between1_idx]+10**(-20))
			m2 = (lim_Z[between2_idx+1] - lim_Z[between2_idx])/(lim_R[between2_idx+1]-lim_R[between2_idx]+10**(-20))

			strikept_dict['leftstrkpt_angle'] = np.abs(np.arctan2(m1-m2, 1+m1*m2)*180.0/np.pi)


	if show_calc == True:
		import matplotlib.pyplot as plt
		from matplotlib import gridspec

		fig1 = plt.figure(figsize=(10,8))
		gs = gridspec.GridSpec(1, 1)#, width_ratios=(35, 1))
		ax1 = plt.subplot(gs[0,0])

		ax1.plot(separx_R, separx_Z, color='k', marker='o', markersize=3)
		ax1.plot(lim_R, lim_Z, color='green', marker='o', markersize=3)

		ax1.plot(Xpt_R, Xpt_Z, color='cyan', marker='D')
		ax1.plot(R_strikes, Z_strikes, color='purple', marker='o',linestyle='none')
		ax1.plot(strikept_dict['left_strkpt'][0], strikept_dict['left_strkpt'][1], color='r', marker='o',linestyle='none')
		ax1.plot(strikept_dict['right_strkpt'][0], strikept_dict['right_strkpt'][1], color='r', marker='o',linestyle='none')
		try:
			ax1.plot(separx_R[between1_idx], separx_Z[between1_idx], color='cyan', marker='s')
			ax1.plot(separx_R[between1_idx+1], separx_Z[between1_idx+1], color='cyan', marker='s')
			ax1.plot(lim_R[between2_idx], lim_Z[between2_idx], color='cyan', marker='s')
			ax1.plot(lim_R[between2_idx+1], lim_Z[between2_idx+1], color='cyan', marker='s')

		except:
			pass

		ax1.set_xlabel('R (m)')
		ax1.set_ylabel('Z (m)')
		#ax1.set_title(gfile_name, loc='Right')
		ax1.minorticks_on()

		ax1.set_aspect('equal')
		fig1.patch.set_facecolor('#ffffff')
		plt.tight_layout(pad=0.06)
		plt.show()


	return strikept_dict

def calc_gaps(R_grid, Z_grid, Psi_arr, Psi_bdry, show_calc=False):
	gaps_dict = OrderedDict() #Initialize a dictionary to hold the coordinates of midplane gaps

	#Coarse scan for the miplane positions of the boundary
	x_interp = np.linspace(np.min(R_grid), np.max(R_grid), 400)
	y_interp = np.zeros(len(x_interp))

	Psi_rad_lin_interp = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr),\
 	(x_interp, y_interp), method='linear')

	if show_calc == True:
		import matplotlib.pyplot as plt
		plt.plot(x_interp, Psi_rad_lin_interp)
		plt.show()
	
	half_idx = np.argmax(Psi_rad_lin_interp)  #Take the first half of the x-coordinates
	x1_est = x_interp[:half_idx][np.argmin(np.abs(Psi_rad_lin_interp[:half_idx]-Psi_bdry))]
	x2_est = x_interp[half_idx:][np.argmin(np.abs(Psi_rad_lin_interp[half_idx:]-Psi_bdry))]
	
	x1_zoom = np.linspace(x1_est-0.02, x1_est+0.02, 401)
	x2_zoom = np.linspace(x2_est-0.02, x2_est+0.02, 401)
	y_zoom = np.zeros(len(x1_zoom))	

	#Now do the fine scan
	Psi_rad_lin_interp1 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr),\
 	(x1_zoom, y_zoom), method='linear')
		
	Psi_rad_lin_interp2 = griddata((np.ravel(R_grid), np.ravel(Z_grid)), np.ravel(Psi_arr),\
 	(x2_zoom, y_zoom), method='linear')

	gaps_dict['midpl_innerR'] = x1_zoom[np.argmin(np.abs(Psi_rad_lin_interp1-Psi_bdry))]
	gaps_dict['midpl_outerR'] = x2_zoom[np.argmin(np.abs(Psi_rad_lin_interp2-Psi_bdry))]
	
	return gaps_dict

#Calculate the 3D angle of incidence for the strike points
def calc_alphas(separx_R, separx_Z, lim_R, lim_Z, gfile_dict0, strikept_dict0, n_phi=0.0, show_calc=False):
	alpha_dict = OrderedDict() #Initialize a dictionary to hold the angle of incidences of the strike points

	#sin(alpha) = b^ dot n^. alpha is the angle of incidence where, b^ = <B_R, B_Z, B_phi>/B_total and n^ = <n_R, n_Z, n_phi> normal to the 
	#target plate (note it can be tilted, yeilding n_phi != 0.0)

	#Calculate B_R and B_Z from the gfile
	dR = np.abs(gfile_dict0['R_grid'][0,0] - gfile_dict0['R_grid'][0,1])
	dZ = np.abs(gfile_dict0['Z_grid'][0,0] - gfile_dict0['Z_grid'][1,0])

	#Take the gradient of the fluxgfile_dict0
	grad_psi = np.gradient(-gfile_dict0['PSIRZ']*(2.0*np.pi), dZ, dR) #2*Pi needed to convert from Wb/Rad to Wb

	B_R = -grad_psi[0]/(2.0*np.pi*gfile_dict0['R_grid'])  
	B_Z = grad_psi[1]/(2.0*np.pi*gfile_dict0['R_grid']) 


	if np.isnan(strikept_dict0['right_strkpt'][0]):
		alpha_dict['rightstrkpt_alpha'] = np.nan
	
	else:
		dist1 = np.sqrt((strikept_dict0['right_strkpt'][0] - separx_R)**2 + (strikept_dict0['right_strkpt'][1] - separx_Z)**2)
		dist1b = np.sqrt((separx_R[1:] - separx_R[:-1])**2 + (separx_Z[1:] - separx_Z[:-1])**2)
		is_between1 = (dist1[1:] + dist1[:-1]) - dist1b
		between1_idx = np.argmin(np.abs(is_between1))
		#dist1_idx = sorted(range(len(dist1)), key=lambda k: dist1[k])[0:2]

		dist2 = np.sqrt((strikept_dict0['right_strkpt'][0] - lim_R)**2 + (strikept_dict0['right_strkpt'][1] - lim_Z)**2) 
		#dist2_idx = sorted(range(len(dist2)), key=lambda k: dist2[k])[0:2]
		dist2b = np.sqrt((lim_R[1:] - lim_R[:-1])**2 + (lim_Z[1:] - lim_Z[:-1])**2)
		is_between2 = (dist2[1:] + dist2[:-1]) - dist2b
		between2_idx = np.argmin(np.abs(is_between2))


		nhat_target = -np.array([lim_Z[between2_idx]-lim_Z[between2_idx+1], -(lim_R[between2_idx]-lim_R[between2_idx+1]),  n_phi])
		nhat_target /= np.linalg.norm(nhat_target)

		#Normal vector in the poloidal plane
		bp_hat = -np.array([separx_R[between1_idx]-separx_R[between1_idx+1],\
		(separx_Z[between1_idx]-separx_Z[between1_idx+1])])
		bp_hat /= np.linalg.norm(bp_hat)

		#Calculate B_R at the strike point
		BR_interp = griddata((np.ravel(gfile_dict0['R_grid']), np.ravel(gfile_dict0['Z_grid'])),\
		np.ravel(B_R), (strikept_dict0['right_strkpt'][0], strikept_dict0['right_strkpt'][1]), method='cubic')
		#Calculate B_Z at the strike point
		BZ_interp = griddata((np.ravel(gfile_dict0['R_grid']), np.ravel(gfile_dict0['Z_grid'])),\
		np.ravel(B_Z), (strikept_dict0['right_strkpt'][0], strikept_dict0['right_strkpt'][1]), method='cubic')

		#Double Check
		#================= 
		#bp_hat2 = np.array([BR_interp, BZ_interp])
		#bp_hat2 /= np.linalg.norm(bp_hat2)

		#print('bp_hat', bp_hat)
		#print('bp_hat2', bp_hat2)
		#=================
		b_hat = np.zeros(3)
		b_hat[0:2] = bp_hat*np.sqrt((BR_interp**2) + (BZ_interp**2))
		B_phi = gfile_dict0['RCENTR']*gfile_dict0['BCENTR']/strikept_dict0['right_strkpt'][1]
		b_hat[2] = B_phi 
		b_hat /= np.linalg.norm(b_hat)


		alpha_dict['rightstrkpt_alpha'] = np.sin(np.abs(np.dot(b_hat, nhat_target)))*180.0/np.pi

		if show_calc == True:
			import matplotlib.pyplot as plt
			plt.plot(lim_R, lim_Z)
			plt.plot(separx_R, separx_Z)
			plt.plot(separx_R[between1_idx], separx_Z[between1_idx], marker='o', color='k')
			plt.plot(separx_R[between1_idx+1], separx_Z[between1_idx+1], marker='o', color='k')

			plt.plot(lim_R[between2_idx], lim_Z[between2_idx], marker='o', color='r')
			plt.plot(lim_R[between2_idx]+nhat_target[0]/40.0, lim_Z[between2_idx]+nhat_target[1]/40.0, marker='o', color='cyan')
			plt.plot(lim_R[between2_idx+1], lim_Z[between2_idx+1], marker='o', color='r')

			plt.gca().set_aspect('equal')
			plt.show()

	if np.isnan(strikept_dict0['left_strkpt'][0]):
		alpha_dict['leftstrkpt_alpha'] = np.nan
	
	else:
		dist1 = np.sqrt((strikept_dict0['left_strkpt'][0] - separx_R)**2 + (strikept_dict0['left_strkpt'][1] - separx_Z)**2)
		dist1b = np.sqrt((separx_R[1:] - separx_R[:-1])**2 + (separx_Z[1:] - separx_Z[:-1])**2)
		is_between1 = (dist1[1:] + dist1[:-1]) - dist1b
		between1_idx = np.argmin(np.abs(is_between1))
		#dist1_idx = sorted(range(len(dist1)), key=lambda k: dist1[k])[0:2]

		dist2 = np.sqrt((strikept_dict0['left_strkpt'][0] - lim_R)**2 + (strikept_dict0['left_strkpt'][1] - lim_Z)**2) 
		#dist2_idx = sorted(range(len(dist2)), key=lambda k: dist2[k])[0:2]
		dist2b = np.sqrt((lim_R[1:] - lim_R[:-1])**2 + (lim_Z[1:] - lim_Z[:-1])**2)
		is_between2 = (dist2[1:] + dist2[:-1]) - dist2b
		between2_idx = np.argmin(np.abs(is_between2))


		nhat_target = -np.array([lim_Z[between2_idx]-lim_Z[between2_idx+1], -(lim_R[between2_idx]-lim_R[between2_idx+1]),  n_phi])
		nhat_target /= np.linalg.norm(nhat_target)

		#Normal vector in the poloidal plane
		bp_hat = -np.array([separx_R[between1_idx]-separx_R[between1_idx+1],\
		(separx_Z[between1_idx]-separx_Z[between1_idx+1])])
		bp_hat /= np.linalg.norm(bp_hat)

		#Calculate B_R at the strike point
		BR_interp = griddata((np.ravel(gfile_dict0['R_grid']), np.ravel(gfile_dict0['Z_grid'])),\
		np.ravel(B_R), (strikept_dict0['left_strkpt'][0], strikept_dict0['left_strkpt'][1]), method='cubic')
		#Calculate B_Z at the strike point
		BZ_interp = griddata((np.ravel(gfile_dict0['R_grid']), np.ravel(gfile_dict0['Z_grid'])),\
		np.ravel(B_Z), (strikept_dict0['left_strkpt'][0], strikept_dict0['left_strkpt'][1]), method='cubic')

		#Double Check
		#================= 
		#bp_hat2 = np.array([BR_interp, BZ_interp])
		#bp_hat2 /= np.linalg.norm(bp_hat2)

		#print('bp_hat', bp_hat)
		#print('bp_hat2', bp_hat2)
		#=================
		b_hat = np.zeros(3)
		b_hat[0:2] = bp_hat*np.sqrt((BR_interp**2) + (BZ_interp**2))
		B_phi = gfile_dict0['RCENTR']*gfile_dict0['BCENTR']/strikept_dict0['left_strkpt'][1]
		b_hat[2] = B_phi
		b_hat /= np.linalg.norm(b_hat)

		if show_calc == True:
			import matplotlib.pyplot as plt
			plt.plot(lim_R, lim_Z)
			plt.plot(separx_R, separx_Z)
			plt.plot(separx_R[between1_idx], separx_Z[between1_idx], marker='o', color='k')
			plt.plot(separx_R[between1_idx+1], separx_Z[between1_idx+1], marker='o', color='k')

			plt.plot(lim_R[between2_idx], lim_Z[between2_idx], marker='o', color='r')
			plt.plot(lim_R[between2_idx]+nhat_target[0]/40.0, lim_Z[between2_idx]+nhat_target[1]/40.0, marker='o', color='cyan')
			plt.plot(lim_R[between2_idx+1], lim_Z[between2_idx+1], marker='o', color='r')

			plt.gca().set_aspect('equal')
			plt.show()


		alpha_dict['leftstrkpt_alpha'] = np.sin(np.abs(np.dot(b_hat, nhat_target)))*180.0/np.pi

	return alpha_dict
