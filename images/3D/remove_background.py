import os

# Uses ImageMagick to remove background from images and crop out fully transparent region.
for image in os.listdir("./"):
    if "isolated" in image:
        print(image)
        os.system(f"convert {image} -transparent '#808080' -transparent '#7F7F7F' -trim +repage {image}")

# mlab.show()
os.system("zenity --info --text 'Finished' --icon-name=emblem-success")
