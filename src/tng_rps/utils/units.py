import astropy.units as u

code_mass = u.def_unit('code_mass', 1.0e10 * u.solMass)
code_length = u.def_unit('code_length', u.kpc)
code_velocity = u.def_unit('code_velocity', u.km / u.s)

standard_mass = u.Msun
standard_length = u.kpc
standard_velocity = u.km / u.s 
standard_time = u.yr
standard_massderivative = u.Msun / u.yr 
standard_volume = (standard_length)**3
standard_energy = u.erg
standard_density = standard_mass / standard_volume
standard_temperature = u.K
standard_metallicity = u.def_unit('Zsun', doc='Solar Metallicity', format=dict(latex=r'Z_{\odot}', latex_inline=r'Z_{\odot}'))
standard_JeansNumber = u.def_unit('Nj',doc='Jeans Number', format=dict(latex=r'N_j', latex_inline=r'N_j'))
standard_pressure = u.erg / u.cm**3
standard_power = u.erg / u.s
standard_entropy = u.keV * u.cm**2
