import onnx
from onnx import checker, GraphProto, TensorProto, AttributeProto, ModelProto
import onnx.numpy_helper
import onnx.defs
import onnx.optimizer
import onnx.shape_inference
import onnx.utils
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict, BackendRep

from onnx.helper import make_tensor_value_info, make_graph, make_model
import numpy as np
import os
import time

import layers
import _backend

class OnnxAttributes(dict):
    @staticmethod
    def from_onnx(args):
        d = OnnxAttributes()
        for arg in args:
            d[arg.name] = convertAttributeProto(arg)
        return d

def convertAttributeProto(onnx_arg):   
    if   onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return onnx_arg.t  # this is a proto!
    elif onnx_arg.HasField('g'):
        return [] #this is graph
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    elif len(onnx_arg.graphs):
        retval = []
        return retval
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        self.inputs = list(node.input)
        self.outputs = list(node.output)
    def bind():
        return (self.inputs, self.outputs, self.attrs)

class OnnxGraph:
    def __init__(self, filename):
        start = time.perf_counter()
        model = onnx.load(filename)
        inferred_model = onnx.shape_inference.infer_shapes(model)
        graph = inferred_model.graph
        self.nodes = list()
        self.layer = list()
        layers.tensors[''] = np.zeros(10)

        for val in inferred_model.graph.value_info:
            data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
            layers.tensors[val.name] = data
        for init_val in graph.initializer:
            name, data = self._create_tensor_filling_op(init_val)            
            layers.tensors[name] = data
            
        for val in graph.input:
            if(val.name not in layers.tensors.keys()):            
                data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
                layers.tensors[val.name] = data     
                
        for val in graph.output:
            if(val.name not in layers.tensors.keys()):
                data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
                layers.tensors[val.name] = data 
                
        for n in graph.node:
            self.nodes.append(OnnxNode(n))
        
        

        _backend.create_instance()

        end = time.perf_counter()
        print("::: DONE MODEL PARSE :::", end-start)



        _backend.test()

    def build(self):
        
        x_start = time.perf_counter()

        start = time.perf_counter()
        for nodes in self.nodes:
            l = layers.layer_map[nodes.op_type](nodes.name, **nodes.attrs)(*nodes.inputs)
            l.output(*nodes.outputs)
            self.layer.append(l)     
        end = time.perf_counter()
        print("::: DONE NODE INIT :::", end-start, "avg", (end-start)/len(self.nodes))

        start = time.perf_counter()
        for name, data in layers.tensors.items():
            _backend.create_tensor(name, data)
        end = time.perf_counter()
        print("::: DONE Tensor BUILD :::", end-start, "avg", (end-start)/len(layers.tensors.items()))

        start = time.perf_counter()
        for l in self.layer:
             l.build()
        end = time.perf_counter()
        print("::: DONE LAYER BUILD :::", end-start, 'avg', (end-start)/len(self.layer))
                
        x_end = time.perf_counter()
        print("::: DONE BUILDING PIPE :::", x_end-x_start)
        
        _backend.test()

    def run(self):

        start = time.perf_counter()
        for layer in self.layer:
            layer.run()
        end = time.perf_counter()

        print("::: DONE RUNNING PIPE :::", end-start, "avg", (end-start)/len(self.layer))

    def _create_tensor_filling_op(self, onnx_tensor, name=None):
      
            assert name or onnx_tensor.name
            name = name or onnx_tensor.name

            def tensor2list(onnx_tensor):
                # Use the onnx.numpy_helper because the data may be raw
                return onnx.numpy_helper.to_array(onnx_tensor).flatten()
            
            if onnx_tensor.data_type == TensorProto.FLOAT:
                pass
            elif onnx_tensor.data_type == TensorProto.DOUBLE:
                pass
            elif onnx_tensor.data_type == TensorProto.INT64:
                pass
            elif onnx_tensor.data_type == TensorProto.UINT32:
                pass
            elif onnx_tensor.data_type == TensorProto.UINT8:
                pass
            elif onnx_tensor.data_type == TensorProto.INT8:
                pass
            elif onnx_tensor.data_type == TensorProto.UINT16:
                pass
            elif onnx_tensor.data_type == TensorProto.INT16:
                pass
            elif onnx_tensor.data_type == TensorProto.INT32:
                pass
            elif onnx_tensor.data_type == TensorProto.BOOL:
                pass
            elif onnx_tensor.data_type == TensorProto.STRING:
                pass
            else:
                raise RuntimeError("unrecognized tensor type {}".format(onnx_tensor.data_type))
    
            if(onnx_tensor.dims == []):
                return (name, tensor2list(onnx_tensor))
            data = tensor2list(onnx_tensor).reshape(*onnx_tensor.dims)
            


            return (name, data)
  
'''
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


'''