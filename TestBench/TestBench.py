import numpy as np
import PIL as Image
import vkFlow as vkFlow

if (__name__ == "__main__"):
    pic = Image.open("ariel.png")
    x = np.array(pic)
    print(x.shape)
   # vkFlow.Run()