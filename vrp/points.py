import random

LAT_MIN, LAT_MAX = -23.685609, -23.480267
LON_MIN, LON_MAX = -46.708364, -46.466336

POINTS = [
    (
        round(random.uniform(LAT_MIN, LAT_MAX), 6),
        round(random.uniform(LON_MIN, LON_MAX), 6)
    )
    for _ in range(300)
]
