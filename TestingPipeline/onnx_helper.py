import onnx
import onnx.backend
from onnx import checker, GraphProto, TensorProto, AttributeProto, ModelProto
import onnx.numpy_helper
import onnx.defs
import onnx.optimizer
import onnx.shape_inference
import onnx.utils
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict
from onnx.helper import make_graph
from onnx.helper import make_tensor
from onnx.helper import make_tensor_value_info
from onnx.helper import mapping

import numpy as np
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

class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        self.inputs = list(node.input)
        self.outputs = list(node.output)
    def bind():
        return (self.inputs, self.outputs, self.attrs)

def _create_tensor_filling_op(onnx_tensor, name=None):
      
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
  

def convertAttributeProto(onnx_arg):   
    if   onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        name, data = _create_tensor_filling_op(onnx_arg.t)
        layers.tensors[name] = data
        return name # this is a proto!
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

class OnnxGraph:
    def __init__(self, filename):       
        _backend.test()  
        self.filename = filename
        start = time.perf_counter_ns() / 1000000
        model = onnx.load(filename)
        pollish_model = onnx.utils.polish_model(model)

        model = onnx.shape_inference.infer_shapes(model)
        
        graph = model.graph

        self.nodes = list()
        self.layer = list()
        self.inputs = list()
        self.outputs = list()
        self.ops = list()

        layers.tensors[''] = np.zeros(10)

        for val in model.graph.value_info:
            data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
            layers.tensors[val.name] = data

        for init_val in graph.initializer:
            name, data = _create_tensor_filling_op(init_val)            
            layers.tensors[name] = data
            
        for val in graph.input:
            if(val.name not in layers.tensors.keys()):            
                data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
                self.inputs.append(val.name)
                layers.tensors[val.name] = data     
                
        for val in graph.output:
            if(val.name not in layers.tensors.keys()):
                data = np.zeros([i.dim_value for i in val.type.tensor_type.shape.dim])
                self.outputs.append(val.name)
                layers.tensors[val.name] = data 
                
        for n in graph.node:
            self.nodes.append(OnnxNode(n))
                          
        for node in self.nodes:
            if(node.op_type not in self.ops):
                self.ops.append(node.op_type)

        end = time.perf_counter_ns() / 1000000
        print(self.filename, "::: DONE MODEL PARSE :::", end-start)        
        x_start = time.perf_counter_ns() / 1000000

        start = time.perf_counter_ns() / 1000000
        for nodes in self.nodes:
            l = layers.layer_map[nodes.op_type](nodes.name, **nodes.attrs)(*nodes.inputs)
            l.output(*nodes.outputs)
            self.layer.append(l)     
        end = time.perf_counter_ns() / 1000000
        print("::: DONE NODE INIT :::", end-start, "avg", (end-start)/len(self.nodes))

        start = time.perf_counter_ns() / 1000000
        for name, data in layers.tensors.items():
            _backend.create_tensor(name, data)
        end = time.perf_counter_ns() / 1000000
        print("::: DONE Tensor BUILD :::", end-start, "avg", (end-start)/len(layers.tensors.items()))

        start = time.perf_counter_ns() / 1000000
        for l in self.layer:
             l.build()
        end = time.perf_counter_ns() / 1000000
        print("::: DONE LAYER BUILD :::", end-start, 'avg', (end-start)/len(self.layer))
        
        x_end = time.perf_counter_ns() / 1000000
        
        print("LAYERS USED: {0}".format(', '.join(self.ops)))

        print("::: DONE BUILDING PIPE :::", x_end-x_start)

    
   
    def __call__(self, *args):
        tmp = None
        for i, x in enumerate(self.inputs):
            _backend.input(x, args[i])
            
        start = time.perf_counter_ns() / 1000000            
        for layer in self.layer:
            layer.run()
            #tmp = _backend.output(layer.name)
        end = time.perf_counter_ns() / 1000000
        print(self.filename, "::: DONE RUNNING PIPE :::", end-start, "avg", (end-start)/len(self.layer))
        
        output = list()
        for y in self.outputs:
            output.append(_backend.output(y))
        return output

    




import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE, TENSOR_TYPE_TO_NP_TYPE
from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from onnx import TensorProto

from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union
from cachetools.func import lru_cache


def onnx_tensor_type_to_numpy_type(data_type):  # type: (Any) -> np.dtype
    if type(data_type) is int:
        return TENSOR_TYPE_TO_NP_TYPE[data_type]
    elif type(data_type) is str:
        return TENSOR_TYPE_TO_NP_TYPE[TensorProto.DataType.Value(data_type)]
    else:
        raise ValueError('Unsupported data type representation (%s).', str(type(data_type)))


def np_dtype_to_tensor_type_name(data_type):  # type: (np.dtype) -> str
    return TensorProto.DataType.Name(NP_TYPE_TO_TENSOR_TYPE[data_type])


def np_dtype_to_tensor_type(data_type):  # type: (np.type) -> int
    return NP_TYPE_TO_TENSOR_TYPE[data_type]

class VulkanBackend(Backend):
    backend_name = 'VKGPU'
    
    @classmethod
    def prepare(cls, onnx_model, device='VKGPU', **kwargs):
        super(VulkanBackend, cls).prepare(onnx_model, device, **kwargs)
        model_import = '' #import_onnx_model(onnx_model)
        reutrn VulkanBackendRep(model_import, cls.backend_name)


    @classmethod
    def run_model(cls, onnx_model, inputs, device='CPU', **kwargs):
        # type: (onnx.ModelProto, List[np.ndarray], str, Dict) -> List[Any]
        """Prepare and run a computation on an ONNX model."""
        return cls.prepare(onnx_model, device, **kwargs).run(inputs)