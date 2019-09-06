import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import onnxruntime as ort
import json
import os
import onnx_helper
import _backend


if(__name__=="__main__"):
    input = np.ones([1,3,1,224,224])
    _backend.create_instance()
    mobilenet = onnx_helper.OnnxGraph('./mobilenetv2.onnx')
    #mnasnet0_5 = onnx_helper.OnnxGraph('./mnasnet0_5.onnx')
    #n2n = onnx_helper.OnnxGraph("./n2n.onnx")
    #bidaf = onnx_helper.OnnxGraph('./bidaf.onnx')  
   
    for _ in range(500):
        mobilenet(input)

    
    #for _ in range(500):
    #   t = mnasnet0_5(input)
    #   t = mobilenet(input)

    #for _ in range(1000):
    #    n2n()
        
    #for _ in range(1000):
    #    bidaf()
    
    import time
    
    sess = ort.InferenceSession('./mobilenetv2.onnx', None)  
    dummy_input = np.random.randn(1,3,224,224).astype(np.float32)
    input_name = sess.get_inputs()[0].name  
    start = time.perf_counter_ns() / 1000000
    output = sess.run(None, {input_name : dummy_input})
    stop = time.perf_counter_ns() / 1000000
    print('onnxruntime ::: ', stop-start)
    