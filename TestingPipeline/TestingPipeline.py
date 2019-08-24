import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_helper
#import _backend as backend
#from layers import *



if(__name__=="__main__"):   
    mobilenet = onnx_helper.OnnxGraph('./mobilenetv2.onnx')
    mobilenet.build()
    print()
    #bidaf = onnx_helper.OnnxGraph('./bidaf.onnx')
    #bidaf.build()
    print()
    