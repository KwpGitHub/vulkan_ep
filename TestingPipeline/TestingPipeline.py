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
    
    print("\n\n")
    for node in graph['node']:
        nodes[node['name']] = node

        name = node['name']
        op_type = node['opType']
        input = node['input']
        output = node['output']
        attribute = {}
        if('attribute' in node.keys()):
            for attr_ in node['attribute']:
                _x = [x for n,x in attr_.items() if(n != 'name' and n != 'type')]
                if(type(_x[0]) == list and len(_x) == 1):
                    attribute[attr_['name']] =  _x[0]
                else:
                    attribute[attr_['name']] = _x
        backend.create_layer(name, op_type, input, output, attribute)
    return nodes


if(__name__=="__main__"):
    backend.test()
    backend.create_instance()
    x = np.ones([1,3,128,128])
    
    onnx_model_str =  MessageToJson(onnx.load('mobilenetv2.onnx'))
    graph = json.loads(onnx_model_str)
    node_info = graph_def_info(graph['graph'])
    backend.input(x)
   
    


  
    