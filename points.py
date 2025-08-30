import random

LAT_MIN, LAT_MAX = -22.416550386078555, -22.383856
LON_MIN, LON_MAX = -47.548246565862065, -47.589960

POINTS = [
    (
        round(random.uniform(LAT_MIN, LAT_MAX), 6),
        round(random.uniform(LON_MIN, LON_MAX), 6)
    )
    for _ in range(300)
]
