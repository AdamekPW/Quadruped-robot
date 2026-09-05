
# Nazwy stóp w modelu Silver Badgera (geomy kolizyjne i site pokrywają się nazwą).
FOOT_SITES = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")
FOOT_GEOMS = ("RL_foot", "RR_foot", "FR_foot", "FL_foot")

# --- Nazwy sensorów w scenie ---
# Trzymane jako stałe, bo odwołują się do nich moduły, które nie budują sensorów:
# nagroda `upright` (lokalne nachylenie terenu), obserwacja CNN i terminacje.
TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"          # z bazy mjlab
FOOT_HEIGHT_SCAN_SENSOR_NAME = "foot_height_scan"  # z bazy mjlab
FEET_CONTACT_SENSOR_NAME = "feet_ground_contact"   # nasz
TRUNK_CONTACT_SENSOR_NAME = "trunk_ground_touch"   # nasz

# Wzorce nazw siłowników NÓG (bez kręgosłupa). Gdy usztywniamy kręgosłup,
# polityka steruje tylko tymi 12 stawami — `spine_joint` zostaje poza akcją.
LEG_ACTUATOR_PATTERNS = (r".*_hip_joint", r".*_thigh_joint", r".*_calf_joint")

# Kształt siatki skanu terenu (height_scan) jako obraz 2D: (H, W).
# Sensor `terrain_scan` to GridPattern size=(1.6, 1.0), resolution=0.1 → rzuty
# generowane meshgridem indexing="xy" i spłaszczane wierszowo. Daje to siatkę
# (oś Y bocznie = 11, oś X do przodu = 17) = 187 punktów. Ten sam wektor,
# poukładany jako (11, 17), odtwarza spójną przestrzennie mapę pod CNN.
#   X: 1.6 m / 0.1 + 1 = 17   (do przodu, wymiar W)
#   Y: 1.0 m / 0.1 + 1 = 11   (bocznie,   wymiar H)
TERRAIN_SCAN_GRID_HW = (11, 17)

# --- Kamera głębi (eksterocepcja STUDENTA, odpowiednik RealSense'a) ---
# Zamontowana z przodu tułowia, patrzy do przodu i w dół (35 stopni), żeby widzieć teren tuż przed nogami
DEPTH_CAMERA_NAME = "front_depth"

# Rozdzielczość kamery
DEPTH_CAMERA_HW = (48, 64)

# Zasięg widzenia [m] - raczej dalej niż 2 m nie jest potrzebne do stawiania kroków.
DEPTH_CUTOFF = 2.0

# Poza względem układu tułowia: 0.28 m do przodu, na wysokości tułowia.
DEPTH_CAMERA_POS = (0.28, 0.0, 0.0)

# (w, x, y, z): patrz do przodu (+x tułowia) z pochyleniem 35 stopni w dół.
DEPTH_CAMERA_QUAT = (0.6272, 0.3265, -0.3265, -0.6272)

# Pionowe pole widzenia [stopnie]
DEPTH_CAMERA_FOVY = 45.0

# --- Podgląd (tryb play) ---
# Ilu robotów sadzamy na każdym polu siatki terenu. Liczba środowisk wynika z
# tego wprost: robots_per_cell * num_rows * liczba typów terenu.
PLAY_ROBOTS_PER_TERRAIN_CELL = 3
