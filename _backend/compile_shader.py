import os
import subprocess


for name in os.listdir("./shaders"):
    subprocess.call("glslangValidator -V -o ./shaders/bin/"+ name.split('.')[0]+'.spv' + " ./shaders/"+name)