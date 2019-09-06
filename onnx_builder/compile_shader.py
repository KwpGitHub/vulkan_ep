import os
import subprocess
for name in os.listdir("../_backend/shaders"):
    if(os.path.isfile("../_backend/shaders/"+name)):
        subprocess.call("glslangValidator -V -o ../_backend/shaders/bin/"+ name.split('.')[0]+'.spv' + " ../_backend/shaders/"+name)