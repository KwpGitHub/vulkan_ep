import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_helper

def graph_def_info(onnx_file):
    onnx_model_str =  MessageToJson(onnx.load(onnx_file))
    graph = json.loads(onnx_model_str)['graph']
    
    backend.create_instance(os.getcwd())
    backend.test()
    
    nodes = {} 
    
    init_nodes = []
    for data in graph['initializer']:
        name    = data['name']
        dims    = [int(x) for x in data['dims']]
             
        init_nodes.append(name)
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
        op_type = node['opType'].lower()
        input = node['input']
        output = node['output']
        layer = layer_map[op_type](name)
        layer.input(*input)
        layer.output(*output)
        if('attribute' in node.keys()):
            for attr_ in node['attribute']:
                _x = [x for n,x in attr_.items() if(n != 'name' and n != 'type')]
                if(type(_x[0]) == list and len(_x) == 1):
                    layer.__dict__[attr_['name']] = _x[0]
                else:
                    layer.__dict__[attr_['name']] = _x
        print(name)
                
    unint_nodes = {}
    for node in graph['input'] + graph['output']:
        if(node['name'] not in init_nodes):
            tmp = np.zeros([int(i['dimValue']) for i in  node['type']['tensorType']['shape']['dim']])
            backend.create_tensor_from_numpy(node['name'], tmp)
    return nodes


if(__name__=="__main__"):   
    mobilenet = onnx_helper.OnnxGraph('./mobilenetv2.onnx')
    mobilenet.build()
    print()
    #bidaf = onnx_helper.OnnxGraph('./bidaf.onnx')
    #bidaf.build()
    print()
    #node_info = graph_def_info('mobilenetv2.onnx')
