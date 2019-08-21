import onnx
from onnx import checker, GraphProto, TensorProto, AttributeProto, ModelProto
import onnx.numpy_helper
import onnx.defs
import onnx.optimizer
import onnx.shape_inference
import onnx.utils
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict

import os
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
        return []
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
        _backend.create_instance(os.getcwd())
        _backend.test()
        model = onnx.load(filename)
        graph = model.graph
        self.nodes = list()
        self.init_vals = dict()
        self.layer = list()
        for init_val in graph.initializer:
            name, data = self._create_tensor_filling_op(init_val)            
            self.init_vals[name] = data
            _backend.create_tensor_from_numpy(name, data)
        for n in graph.node:
            self.nodes.append(OnnxNode(n))
    def build(self):
        for nodes in self.nodes:
            l = layers.layer_map[nodes.op_type](nodes.name)
            l.input(*nodes.inputs)
            l.output(*nodes.outputs)
            l.attribute(**nodes.attrs)
            l.build()
            self.layer.append(l)
            

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
        