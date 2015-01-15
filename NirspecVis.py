import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
import datetime, math, pyfits
from scipy import interpolate 

class ObsEpochSA():
	def __init__(self,EpochLabel, SourceName=None, FitsDir='', Coords=None):
		self.SourceName = SourceName
		self.FitsFilenames = self.MakeFitsFilenames(EpochLabel, FitsDir)
		self.Coords = Coords
		self.BaryVel = self.CalcBaryVel(Coords, EpochLabel)
		self.Data = self.readFits(self.FitsFilenames)
		#self.LineProfiles = None
		self.Catalog = None

	def MakeFitsFilenames(self, EpochLabel, FitsDir):
		name, date, time, orders = EpochLabel[0], EpochLabel[1], EpochLabel[2], EpochLabel[3]
		Norders = len(orders)

		filenames = []
		for i in range(0,Norders):
			filenames.append(FitsDir+name+'_'+date+'_'+time+'_saspec'+str(int(orders[i]))+'.fits')

		return filenames

	def CalcBaryVel(self, Coords, EpochLabel):
		RA, Dec = Coords[0], Coords[1]
		date, time = EpochLabel[1], EpochLabel[2]

		if (Dec[0]<0.0):
			dfac = -1.0
		else:
			dfac = 1.0

		RA_deg  = (15.0*RA[0]) + (RA[1]/4.0) + (RA[2]/240.0)
		Dec_deg = Dec[0] + (dfac*Dec[1]/60.0) + (dfac*Dec[2]/3600.0)

		RA_rad  = RA_deg*(math.pi/180.0)
		Dec_rad = Dec_deg*(math.pi/180.0)

		year, month, day = int(date[0:4]), int(date[4:6]), int(date[6:8])
		hour, minute = int(time[0:2]), int(time[2:4])

		DT_obj = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
		JulDay = pyasl.jdcnv(DT_obj)

		HelioVel, BaryVel = pyasl.baryvel(JulDay,0)

		BaryVel_obs = BaryVel[0]*math.cos(Dec_rad)*math.cos(RA_rad) + BaryVel[1]*math.cos(Dec_rad)*math.sin(RA_rad) + BaryVel[2]*math.sin(Dec_rad)
		BaryVel_cor = 1.0*BaryVel_obs

		return BaryVel_cor

	def readFits(self, FitsFilenames):
		norders = len(FitsFilenames)

		FitsInfo = []
		for i in range(0,norders):
			hdulist = pyfits.open(FitsFilenames[i])
			data = hdulist[1].data
			npix = data.shape[0]

			wl = np.array([])
			flux, uflux = np.array([]), np.array([])
			sa, usa = np.array([]), np.array([])

			for data_oi in data:
				wl  = np.append(wl, data_oi[0])
				flux, uflux = np.append(flux, data_oi[1]), np.append(uflux, data_oi[2])
				sa, usa     = np.append(sa, data_oi[3]), np.append(usa, data_oi[4])

			FitsInfo.append({'WLdata':wl, 'Fdata':flux, 'UFdata':uflux, 'SAdata':sa, 'USAdata':usa})

		return FitsInfo

	def MakeLineProfiles(self, VelGrid, TransForAvg, MolName, WinMin, WinMax):
		cc = 3.0e5
		
		norders = len(self.Data)
		NumForAvg = len(TransForAvg)
		
		WLCat, MolCat, TransCat = self.Catalog['WLCat'], self.Catalog['MolCat'], self.Catalog['TransCat']
		NumCatTrans = len(WLCat)

		SAGrid, SAGridSub, USAGrid = [], [], []
		FluxGrid, FluxGridNorm, UFluxGrid, UFluxGridNorm = [], [], [], []
		InData = []
		for i in range(0,NumForAvg):
			indx = np.where(TransCat == TransForAvg[i])
			WLCat_oi, TransCat_oi = WLCat[indx], TransCat[indx]
			InData.append(False)

			for j in range(0,norders):
				data = self.Data[j]
				wl, flux, uflux = data['WLdata'], data['Fdata'], data['UFdata']
				sa, usa = data['SAdata'], data['USAdata']

				if (WLCat_oi > min(wl) and WLCat_oi < max(wl)):
					InData[i] = True
					vel = np.multiply(np.divide(np.subtract(wl,WLCat_oi), WLCat_oi), cc)

					FuncGridF   = interpolate.interp1d(vel, flux, bounds_error=False)
					FuncGridUF  = interpolate.interp1d(vel, uflux, bounds_error=False)
					FuncGridSA  = interpolate.interp1d(vel, sa, bounds_error=False)
					FuncGridUSA = interpolate.interp1d(vel, usa, bounds_error=False)

					sa_grid, usa_grid = FuncGridSA(VelGrid), FuncGridUSA(VelGrid)
					flux_grid, uflux_grid = FuncGridF(VelGrid), FuncGridUF(VelGrid)

					sa_sub = BaseSubtract(VelGrid, sa_grid, WinMin, WinMax)
					flux_norm, uflux_norm = ContNorm(VelGrid, flux_grid, uflux_grid, WinMin, WinMax)

					SAGrid.append(sa_grid), SAGridSub.append(sa_sub), USAGrid.append(usa_grid)
					FluxGrid.append(flux_grid), FluxGridNorm.append(flux_norm)
					UFluxGrid.append(uflux_grid), UFluxGridNorm.append(uflux_norm)
					break

			if (InData[i] == False):
				SAGrid.append(None), SAGridSub.append(None), USAGrid.append(None)
				FluxGrid.append(None), FluxGridNorm.append(None)
				UFluxGrid.append(None), UFluxGridNorm.append(None)

		output = {'TransForAvg':TransForAvg, 'InData':InData, 'VelGrid':VelGrid, 'SAGrid':SAGrid, 'SAGridSub':SAGridSub, 'USAGrid':USAGrid,
				  'FluxGrid':FluxGrid, 'FluxGridNorm':FluxGridNorm, 'UFluxGrid':UFluxGrid, 'UFluxGridNorm':UFluxGridNorm}

		return output
		#self.LineProfiles = output

	def ReadCat(self, filename):
		CatInfo = np.loadtxt(filename, dtype='str')
		wl    = np.array(CatInfo[:,0], dtype='float')
		mol   = np.array(CatInfo[:,1], dtype='str')
		trans = np.array(CatInfo[:,2], dtype='str')

		self.Catalog = {'WLCat':wl, 'MolCat':mol, 'TransCat':trans}

	def getData(self):
		return self.Data

	def getSourceName(self):
		return self.SourceName


############################
############################


def CompAvgLineProfile(LineProfiles, yname_input):
	yname, uyname = PickYName(yname_input)

	TransForAvgInit, InData = np.array(LineProfiles['TransForAvg']), np.array(LineProfiles['InData'])
	YvecInit, UYvecInit = np.array(LineProfiles[yname]), np.array(LineProfiles[uyname])
	VelGrid = np.array(LineProfiles['VelGrid'])
	Npix = len(VelGrid)

	SecInData = np.where(InData == True)
	TransForAvg, Yvec, UYvec = TransForAvgInit[SecInData], YvecInit[SecInData], UYvecInit[SecInData]
	NumLines = len(TransForAvg)

	YvecAvg, UYvecAvg = np.array([]), np.array([])
	for i in range(0, Npix):
		YvecO_nan, UYvecO_nan = np.array([]), np.array([])
		for j in range(0,NumLines):
			yvec, uyvec = Yvec[j], UYvec[j]
			YvecO_nan, UYvecO_nan = np.append(YvecO_nan, yvec[i]), np.append(UYvecO_nan, uyvec[i])
		
		SecNoNan = np.logical_not(np.isnan(YvecO_nan))
		YvecO, UYvecO = YvecO_nan[SecNoNan], UYvecO_nan[SecNoNan]

		if (len(YvecO) == 0):
			YvecAvg, UYvecAvg = np.append(YvecAvg, np.nan), np.append(UYvecAvg, np.nan)
		else:
			#uncertainty average needs to be edited here
			YvecAvg, UYvecAvg = np.append(YvecAvg, np.average(YvecO)), np.append(UYvecAvg, np.average(UYvecO))

	return YvecAvg, UYvecAvg


############################
############################

def PrintLines(self):
	trans = self.LineProfiles['TransForAvg']
	in_data = self.LineProfiles['InData']
	ntrans = len(trans)

	for i in range(0,ntrans):
		print trans[i], in_data[i]

############################
############################

def PickYName(yname_input):
	if (yname_input == 'SA'):
		yname_output, uyname_output = 'SAGrid', 'USAGrid'

	if (yname_input == 'SA_sub'):
		yname_output, uyname_output = 'SAGridSub', 'USAGrid'

	if (yname_input == 'Flux'):
		yname_output, uyname_output = 'FluxGrid', 'UFluxGrid'

	if (yname_input == 'Flux_norm'):
		yname_output, uyname_output = 'FluxGridNorm', 'UFluxGridNorm'

	return yname_output, uyname_output

############################
############################

def BaseSubtract(vel, yvec, win_min, win_max):
	sec = np.where((vel < win_min) | (vel > win_max))
	bg = yvec[sec]
	base = np.median(bg[np.logical_not(np.isnan(bg))])
	yvec_new = np.subtract(yvec, base)
	
	return yvec_new

############################
############################

def ContNorm(vel, yvec, uyvec, win_min, win_max):
	sec = np.where((vel < win_min) | (vel > win_max))
	bg = yvec[sec]
	base = np.median(bg[np.logical_not(np.isnan(bg))])
	yvec_new = np.divide(yvec, base)
	uyvec_new = np.divide(uyvec, base)

	return yvec_new, uyvec_new

############################
############################

def MakeHistVecs(xvec, yvec):
	npix = len(xvec)

	xvec_bin, yvec_bin = np.array([]), np.array([])
	for i in range(0,npix):
		if (i == 0):
			dx_up = dx_low = (xvec[i+1] - xvec[i]) / 2.0
		elif (i == npix-1):
			dx_up = dx_low = (xvec[i] - xvec[i-1]) / 2.0
		else:
			dx_up  = (xvec[i+1] - xvec[i]) / 2.0
			dx_low = (xvec[i] - xvec[i-1]) / 2.0

		xvec_bin = np.append(xvec_bin, (xvec[i]-dx_low, xvec[i]+dx_up))
		yvec_bin = np.append(yvec_bin, (yvec[i], yvec[i]))

	xvec_out, yvec_out = np.ravel(xvec_bin), np.ravel(yvec_bin)

	return xvec_out, yvec_out




