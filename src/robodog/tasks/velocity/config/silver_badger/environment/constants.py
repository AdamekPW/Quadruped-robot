
# Nazwy stóp w modelu Silver Badgera (geomy kolizyjne i site pokrywają się nazwą).
FOOT_SITES = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")
FOOT_GEOMS = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")

# Wzorce nazw siłowników NÓG (bez kręgosłupa). Gdy usztywniamy kręgosłup,
# polityka steruje tylko tymi 12 stawami — `spine_joint` zostaje poza akcją.
LEG_ACTUATOR_PATTERNS = (r".*_hip_joint", r".*_thigh_joint", r".*_calf_joint")