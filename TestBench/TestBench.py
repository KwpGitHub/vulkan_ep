import numpy as np
from PIL import Image
import vkFlow as vkFlow
import onnx

if (__name__ == "__main__"):
    onnx_model = onnx.load('mobilenetv2.onnx')
    img = Image.open("aerial.png")
    x = np.array(img)
    print(x.shape)
   # vkFlow.Run()