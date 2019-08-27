import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_helper



if(__name__=="__main__"):   
    mobilenet = onnx_helper.OnnxGraph('./mobilenetv2.onnx')
    mobilenet.build()
    mobilenet.run()
    print()
    #bidaf = onnx_helper.OnnxGraph('./bidaf.onnx')
    #bidaf.build()
    print()
    