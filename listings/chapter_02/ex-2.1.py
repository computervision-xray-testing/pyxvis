from pyxvis.io import gdxraydb
from pyxvis.io.visualization import show_series

gdxraydb.xgdx_stats()

image_set = gdxraydb.Baggages()

image_set.describe()

print(image_set.get_dir(60))

try:
    series_dir = image_set.get_dir(83)
except ValueError as err:
    print(err)

show_series(image_set, 8, range(1, 352, 10), n=18, scale=0.2)
