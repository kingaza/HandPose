import os
import cv2   
   
poses = os.listdir('Poses/')
for pose in poses:
    print(">> Working on pose : " + pose)
    subdirs = os.listdir('Poses/' + pose + '/') 
    for subdir in subdirs:
        files = os.listdir('Poses/' + pose + '/' + subdir + '/')
        print(">> Working on examples : " + subdir)
        for file in files:
            if(file.endswith(".png")):
                path = 'Poses/' + pose + '/' + subdir + '/' + file
                # Read image
                im = cv2.imread(path)

                im = cv2.flip(im, 1)

                cv2.imwrite(path, im)
