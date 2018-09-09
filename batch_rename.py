import os
path = './resources/real/'

for filename in os.listdir(path):
    num = filename[:-4]
    num = num.zfill(4)
    new_filename = num + ".png"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))