'''
Series of magnitude distributions to be included in the catalog
'''


#class Uniform:

class AbsolutePowerLaw:
	def __init__(self, slope, H_break = None):
		self.type = 'absolute'
#class AbsolutePowerLawWithBreak:

class ApparentPowerLaw:
	def __init__(self, slope, m_break = None):
		self.type = 'apparent'

#class ApparentPowerLawWithBreak:
