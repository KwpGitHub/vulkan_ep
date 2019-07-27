import os
import subprocess

lst = os.listdir("./shaders")
for name in os.listdir("./shaders"):
    if(os.path.isfile("./shaders/"+name)):
        subprocess.call("glslangValidator -V -o ./shaders/bin/"+ name.split('.')[0]+'.spv' + " ./shaders/"+name)