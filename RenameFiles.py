import os

#=================================#
# walk through all files in a directory
# remove the word "Manning." in the name of each file
# set new name for each file without "Manning."
#=================================#

dir = "some folder"
for filename in os.listdir(dir):
    if filename.__contains__("Manning."):
        print(filename)
        newFileName = filename.replace("Manning.", "")
        #newFileName = "Manning." + filename
        os.rename(dir + filename, dir + newFileName)
        print(newFileName)


