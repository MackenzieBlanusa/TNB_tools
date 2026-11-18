"""Variable naming and metadata conventions for L1 mooring files.

Edit this file (not the notebooks) when a new instrument brings a new variable name.
"""

# Instrument-specific name -> standard name.
# Any variable not listed here and not in STANDARD_VARS is kept as is, with a warning.
RENAME = {
    # CTD
    "T": "temp",
    "S": "sal",
    "P": "pres",
    "dep": "depth",
    "sig_t": "sigma_t",
    "Density": "density",
    # current meters
    "spe": "speed",
    "dirt": "dir",
    "hdgt": "heading",
    "A1": "amp1",
    "A2": "amp2",
    "A3": "amp3",
    "volt": "battery",
    "bv": "battery",
    "sound speed": "sound_speed",
    "sound speed used": "sound_speed_used",
    "analog 1": "analog1",
    "analog 2": "analog2",
}

# Standard names with the units and long_name every L1 file will carry.
STANDARD_VARS = {
    "temp": ("degree_Celsius", "In-situ temperature (ITS-90)"),
    "sal": ("1", "Practical salinity (PSS-78)"),
    "con": ("S m-1", "Conductivity"),
    "pres": ("dbar", "Sea pressure"),
    "depth": ("m", "Depth from pressure"),
    "sigma0": ("kg m-3", "Potential density anomaly, ref 0 dbar"),
    "sigma_t": ("kg m-3", "Density anomaly sigma-t"),
    "density": ("kg m-3", "In-situ density"),
    "abs_sal": ("g kg-1", "Absolute salinity (TEOS-10)"),
    "cons_temp": ("degree_Celsius", "Conservative temperature (TEOS-10)"),
    "theta": ("degree_Celsius", "Potential temperature, ref 0 dbar"),
    "temp_freezing": ("degree_Celsius", "In-situ freezing temperature"),
    "cons_temp_freezing": ("degree_Celsius", "Conservative freezing temperature"),
    "u": ("cm s-1", "Eastward velocity"),
    "v": ("cm s-1", "Northward velocity"),
    "w": ("cm s-1", "Upward velocity"),
    "speed": ("cm s-1", "Current speed"),
    "dir": ("degree", "Current direction (toward), clockwise from true north"),
    "heading": ("degree", "Instrument heading"),
    "pitch": ("degree", "Instrument pitch"),
    "roll": ("degree", "Instrument roll"),
    "tilt": ("degree", "Instrument tilt"),
    "amp1": ("counts", "Beam 1 amplitude"),
    "amp2": ("counts", "Beam 2 amplitude"),
    "amp3": ("counts", "Beam 3 amplitude"),
    "battery": ("V", "Battery voltage"),
    "sound_speed": ("m s-1", "Sound speed"),
    "sound_speed_used": ("m s-1", "Sound speed used by instrument"),
    "analog1": ("counts", "Analog input 1"),
    "analog2": ("counts", "Analog input 2"),
}

# Velocity components that must end up in cm/s.
VELOCITY_VARS = ("u", "v", "w")

# Instrument types used in file names and the catalog.
INSTRUMENT_TYPES = {
    "ctd": "Conductivity-temperature-pressure recorder (e.g. SBE37)",
    "temp": "Temperature logger (e.g. SBE56)",
    "vel": "Acoustic point current meter (Nortek Aquadopp)",
    "rcm": "Rotor current meter (Aanderaa RCM9); u, v computed from speed and direction",
}

# Instrument model (as in the raw files' description attribute) -> L1 type code.
MODEL_TYPE = {"SBE37SM": "ctd", "SBE56": "temp", "AQD": "vel", "RCM9": "rcm"}

# Legacy file-name type codes -> L1 type code.
# Legacy 'speed' files are rotor current meters (speed + dir only).
LEGACY_TYPE = {"ctd": "ctd", "temp": "temp", "vel": "vel", "speed": "rcm"}

# Default packing for L1 files: float32 + compression roughly halves to quarters file size.
ENCODING_FLOAT = {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": float("nan")}
