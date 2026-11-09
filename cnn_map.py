import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import netCDF4 as nc
import cartopy.crs as ccrs  # For better handling of geographical data
import cartopy.feature as cfeature  # To add geographical features
import cartopy.io.img_tiles as cimgt

sst_data = nc.Dataset("sst.mon.mean.trefadj.anom.1880to2018.nc")
print(sst_data)
sst = sst_data.variables['sst'][564,:,:]
lon = sst_data.variables['lon'][:]
lat = sst_data.variables['lat'][:]
lon, lat = np.meshgrid(lon, lat)

plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([90, 160, 5, 55], crs=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.LAND, facecolor='none')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# Plot vorticity as filled contours
levels=np.arange(-3,3)
cs_sp = ax.contourf(lon, lat,sst,camp = 'coolwarm',levels=levels,extend='max',transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs_sp, ax=ax)
cbar.set_label('SST')

# Overlay geopotential height as contour lines
gl = ax.gridlines(draw_labels=True, color='gray', linestyle='--', linewidth=0.1)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

plt.xlabel('Longitude')
plt.ylabel('Latitude')


# Customize and save the plot
plt.title('1960-01-01')
plt.savefig('hw9-0.png',dpi=300)
plt.show()
