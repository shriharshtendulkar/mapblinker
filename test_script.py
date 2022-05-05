from mapblinker import MapBlinker

path = "/Users/shriharsh/Documents/Work/AllSkyArray/GMRT_RFI/"

file1 = "Cartosat_images/221583911/December2020_merged.tif"
file2 = "Cartosat_images/221584011/January2021_merged.tif"

mb = MapBlinker(path + file1, path + file2)

mb.run_display_loop()
