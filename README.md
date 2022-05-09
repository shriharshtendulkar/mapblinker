# mapblinker
Blink comparison of satellite imagery

Requirements:
1) rasterio
2) pyproj
3) numpy
4) matplotlib

## Installation instructions
1. Install miniconda (https://docs.conda.io/en/latest/miniconda.html). Choose the 64-bit Python 3.8 or 3.9 and the appropriate operating system. Follow the instructions to create a base environment and activate it. Make sure you install conda in a location where you have permissions to read and update the packages.
2. Create a new conda environment. Lets call it `geotiff`. `conda create -name geotiff`
3. `conda deactivate`
4. `conda activate geotiff`
5. Install `rasterio` `pyproj` and other things: 

`conda config --add channels conda-forge`

`conda config --set channel_priority strict`

`conda install -c conda-forge gdal -n geotiff` 

`conda install pyproj -n geotiff`

`conda install rasterio -n geotiff`

`conda install numpy ipython matplotlib -n geotiff`

6. More detailed instructions here https://rasterio.readthedocs.io/en/latest/installation.html, https://pyproj4.github.io/pyproj/stable/installation.html,  
7. Clone the `mapblinker` git repo: `git clone git@github.com:shriharshtendulkar/mapblinker.git`
8. Copy the data (Email me for the link)
9. Go to the mapblinker directory. Edit `test_script.py` so that the file paths are pointed to the correct location. Save the changes.
10. Run the code in the terminal: Execute `python test_script.py` in the same directory as `mapblinker.py`
11. The code will print some statistics and numbers on the screen and after a while a matplotlib window will show up. If you click at a location, the coordinates of the location will be printed in the terminal. A snapshot image of the surrounding 100m x 100m is also saved in the same directory. 
12. If you type `n` or `N`, you'll go to the next image. If you type `q` or `Q`, the program will quit.
