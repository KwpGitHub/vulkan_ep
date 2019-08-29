import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_helper
import _backend


if(__name__=="__main__"):
    input = np.ones([1,3,1,244,244])
    _backend.create_instance()


    mobilenet = onnx_helper.OnnxGraph('./mobilenetv2.onnx')
    mnasnet0_5 = onnx_helper.OnnxGraph('./mnasnet0_5.onnx')
    #n2n = onnx_helper.OnnxGraph("./n2n.onnx")
    #bidaf = onnx_helper.OnnxGraph('./bidaf.onnx')
   
    for _ in range(1000):
        mobilenet()    
    print()
    
    for _ in range(500):
        mnasnet0_5()
        mobilenet()

    #for _ in range(1000):
    #    n2n()
        
   #for _ in range(1000):
   #    bidaf()
    print()
    