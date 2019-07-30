import numpy as np
import _backend as backend

from google.protobuf.json_format import MessageToJson
import onnx
import json
import os

def graph_def_info(graph):
    nodes = {}
   
    
    for data in graph['initializer']:
        name    = data['name']      
        dims    = [int(x) for x in data['dims']]
        if(data['dataType'] == 7):
            data_t = [float(x) for x in data['int64Data']]
            backend.create_tensor(name, data_t, dims)
        elif(data['dataType'] == 1):
            data_t  = [float(x) for x in data['floatData']]
            backend.create_tensor(name, data_t, dims)
        else:
            print(data['name'], data['dataType'])
      
    for node in graph['node']:
        nodes[node['name']] = node
        for i, input in enumerate(node['input']):
            for data_names in init_vals.keys():
                if(data_names == input):
                    nodes[node['name']]['input'][i] = init_vals[data_names]


if(__name__=="__main__"):
    x = np.ones(128)
    #backend.test()
    backend.create_instance()
  
    onnx_model_str =  MessageToJson(onnx.load('mobilenetv2.onnx'))
    graph = json.loads(onnx_model_str)
    node_info = graph_def_info(graph['graph'])
    backend.input(x)
    


  
    