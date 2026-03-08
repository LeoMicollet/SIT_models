import math

params = {
    "birth": 10 / 64,  # Birth rate per adult
    "n_egg": 64,
    "mu": 0.5,  # Rate at which egg becomes a female
    "egg_death": 0.046,  # Death rate per egg
    "larva_death": 0.05,  # Death rate per larva
    "pupa_death": 0.005,  # Death rate per pupa
    "female_death": 0.046,  # Death rate per adult female
    "male_death": 0.139,  # Death rate per adult male
    "sterile_death": 0.139 * 1.2,  # Death rate per sterile male
    "c": 0.001,  # Competition coefficient for larvae
    "transi_el": 0.79,
    "transi_lp": 0.125,
    "transi_pa": math.log(2) / 2,  # Transition rate from pupa to adult
    "t": 0.0,
    "transiL_mod": 0.125 * (math.log(2) / 2) / (0.005 + (math.log(2) / 2)),
    "deltaA": 0.046,
    "deltaE": 0.046,
    "death_L": 0.05,
    "death_P": 0.005,
    "transi_mod": 0.125 * (math.log(2) / 2) / (0.005 + (math.log(2) / 2)),
}