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

    


from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union

def force_unicode(s):
     try:
         return s.decode('utf-8')
     except AttributeError:
         return s

def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU, DeviceType.CUDA: workspace.GpuDeviceType}
    return core.DeviceOption(m[device.type], device.device_id)
 

class OnnxAttributes(dict):
    @staticmethod
    def from_onnx(args):
        d = OnnxAttributes()
        for arg in args:
            d[arg.name] = convertAttributeProto(arg)
        return d



def convertAttributeProto(onnx_arg):   
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return onnx_arg.t
    elif onnx_arg.HasField('g'):
        return VulkanBackend._graph_to_net(onnx_arg.g, VulkanBackend._known_opset_version)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    elif len(onnx_arg.graphs):
        retval = []
        for g in onnx_arg.graphs:
            retval.append(VulkanBackend._graph_to_net(g, VulkanBackend._known_opset_version))
        return retval
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class VulkanBackendRep(BackendRep):
    def __init__(self, graph=None, inputs=None, outputs=None, tensor_dict=None):
        super(VulkanBackendRep, self).__init__()
        self._graph = graph
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._tensor_dict = tensor_dict or {}
    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def tensor_dict(self):
        return self._tensor_dict

    @tensor_dict.setter
    def tensor_dict(self, tensor_dict):
        self._tensor_dict = tensor_dict

    def run(self, inputs, **kwargs):
        super(VulkanBackendRep, self).run(inputs, **kwargs)


class VulkanBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(VulkanBackend, cls).prepare(model, device, **kwargs)
        return cls.onnx_model_to_rep(model)
    
    @classmethod
    def onnx_model_to_tensorflow_rep(cls, model):
        return cls._onnx_graph_to_tensorflow_rep(model.graph)

    def _onnx_graph_to_tensorflow_rep(cls, graph_def, opset, strict):
        if graph_def.initializer:
            input_dict_items = cls._onnx_initializer_to_input_dict_items(graph_def.initializer)
            initialized = {init.name for init in graph_def.initializer}
        else:
            input_dict_items = []
            initialized = set()
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue
            shape = list( d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None for d in value_info.type.tensor_type.shape.dim)
            input_dict_items.append((value_info.name, x))
        tensor_dict = dict(input_dict_items)
        input_dict = dict(input_dict_items)
        for node in graph_def.node:
            onnx_node = OnnxNode(node)
            output_ops = cls._onnx_node_to_tensorflow_op(onnx_node, tensor_dict, handlers, opset=opset, strict=strict)
            curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
            tensor_dict.update(curr_node_output_map)
        tf_rep = VulkanBackendRep()
        tf_rep.graph = tf_rep_graph
        tf_rep.inputs = [value_info.name for value_info in graph_def.input if value_info.name not in initialized ]
        tf_rep.outputs = [value_info.name for value_info in graph_def.output]
        tf_rep.tensor_dict = tensor_dict
        return tf_rep

    @classmethod
    def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
        super(VulkanBackend, cls).run_node(node, inputs, device)

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer):

        def tensor2list(onnx_tensor):
            return numpy_helper.to_array(onnx_tensor).flatten().tolist()

        return [(init.name, tensor2list(init).reshape(*init.dims)) for init in initializer ]
    @classmethod
    def _onnx_node_to_tensorflow_op(cls, node, tensor_dict, handlers=None,):
        pass

class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.
    """

    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name, attr_translator.translate_onnx(attr.name, attr_converter.convert_onnx(attr))) for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node


class OnnxGraph(object):
    def __init__(self, name=None, graph_proto=None):
        if graph_proto:
            self._name = graph_proto.name
            self._inputs_proto = list(graph_proto.input)
            self._outputs_proto = list(graph_proto.output)
            self._nodes_proto = list(graph_proto.node)
            self._consts_proto = list(graph_proto.initializer)
            self._value_info_proto = list(graph_proto.value_info)
            self._consts = dict([(init.name, numpy_helper.to_array(init))
                                for init in graph_proto.initializer])
        else:
            self._name = name or ""
            self._inputs_proto = []
            self._outputs_proto = []
            self._nodes_proto = []
            self._consts = {}
            self._consts_proto = []
            self._value_info_proto = []
    
    self._data_type_cast_map = {}

    self._add_utility_constants()

    def _add_utility_constants(self):
        util_consts = {CONST_ONE_FP32: np.array([1.0]).astype(np.float32)}
        # Add a few useful utility constants:
        for name, value in util_consts.items():
            self.add_const_explicit(name=name, value=value)
            self.add_const_proto_explicit(
                name=name, value=value, np_dtype=value.dtype)
            self.add_input_proto_explicit(
                name=name, shape=value.shape, np_dtype=value.dtype)

    # This list holds the protobuf objects of type ValueInfoProto
    # representing the input to the converted ONNX graph.
    @property
    def inputs_proto(self):
        return self._inputs_proto

    @inputs_proto.setter
    def inputs_proto(self, inputs_proto):
        self._inputs_proto = inputs_proto

    @property
    def all_node_inputs(self):
        return list(chain.from_iterable(map(lambda p: p.input, self._nodes_proto)))

    @property
    def outputs(self):
        return list(map(lambda p: p.name, self._outputs_proto))

    @property
    def outputs_proto(self):
        return self._outputs_proto

    # This list holds the protobuf objects of type NodeProto
    # representing the ops in the converted ONNX graph.
    @property
    def nodes_proto(self):
        return self._nodes_proto

    @nodes_proto.setter
    def nodes_proto(self, nodes_proto):
        self._nodes_proto = nodes_proto

   
    @property
    def consts(self):
        return self._consts

    @consts.setter
    def consts(self, consts):
        self._consts = consts

    @property
    def consts_proto(self):
        return self._consts_proto

    @consts_proto.setter
    def consts_proto(self, consts_proto):
        self._consts_proto = consts_proto

  
    @property
    def data_type_cast_map(self):
        return self._data_type_cast_map

    @data_type_cast_map.setter
    def data_type_cast_map(self, data_type_cast_map):
        self._data_type_cast_map = data_type_cast_map

    
    @property
    def value_info_proto(self):
        return self._value_info_proto

    def add_input_proto_explicit(self, name, shape, np_dtype=None, tf_dtype=None, onnx_dtype=None):
        onnx_dtype = any_dtype_to_onnx_dtype(np_dtype=np_dtype, tf_dtype=tf_dtype, onnx_dtype=onnx_dtype)
        input_proto = make_tensor_value_info(name, onnx_dtype, shape)
        self._inputs_proto.append(input_proto)

    def add_input_proto(self, node):
        name = node.name
        onnx_dtype = node.attr["dtype"]
        shape = node.attr["shape"] if node.op_type != "Const" else node.attr['value'].shape
        self.add_input_proto_explicit(name, shape, onnx_dtype=onnx_dtype)

    def add_output_proto(self, node):
        output_onnx_type = node.attr.get("T", TensorProto.BOOL)
        for i, output_shape in enumerate(node.attr["_output_shapes"]):
            output_name = node.name + ":{}".format(i) if i > 0 else node.name
            self._outputs_proto.append(
                make_tensor_value_info(output_name, output_onnx_type, output_shape))

    def add_node_proto(self, node_proto):
        if not isinstance(node_proto, (list, tuple)):
            node_proto = [node_proto]
        self._nodes_proto.extend(node_proto)

    def remove_node_proto(self, names):
        if not isinstance(names, (list, tuple)):
            names = [names]
        self._nodes_proto = list(
            filter(lambda x: x.name not in names, self._nodes_proto))

    def add_const_explicit(self, name, value):
        self._consts[name] = value

    def add_const(self, node):
        self.add_const_explicit(node.name, node.attr["value"])

    def add_const_proto_explicit(self, name, value, np_dtype=None, tf_dtype=None, onnx_dtype=None):
        onnx_dtype = any_dtype_to_onnx_dtype(np_dtype=np_dtype, tf_dtype=tf_dtype, onnx_dtype=onnx_dtype)
        const_dim = len(value.shape)

        if const_dim == 0:
            raw_values = [value.tolist()]
            values = [value]
        else:
            raw_values = value.flatten().tolist()
            values = value

        shape = np.array(values).shape
        const_proto = make_tensor(name=name, data_type=onnx_dtype, dims=shape, vals=raw_values)
        self._consts_proto.append(const_proto)

    def add_const_proto(self, node):
        self.add_const_proto_explicit(node.name, node.attr["value"], onnx_dtype=node.attr["dtype"])

    def add_value_info_proto(self, node):
        node_onnx_type = node.attr.get("T", TensorProto.BOOL)
        for i, output_shape in enumerate(node.attr["_output_shapes"]):
            node_name = node.name + ":{}".format(i) if i > 0 else node.name
            value_info_proto = make_tensor_value_info(node_name, node_onnx_type,
                                                    output_shape)
            self._value_info_proto.append(value_info_proto)

  
    def _clean_graph(self):
        in_out = self.all_node_inputs + self.outputs
        self._inputs_proto = list(
            filter(lambda x: x.name in in_out, self.inputs_proto))
        self._consts_proto = list(
            filter(lambda x: x.name in in_out, self.consts_proto))

    def _fix_data_type(self):
        self.inputs_proto = self._data_type_caster(self.inputs_proto,
                                                    self.data_type_cast_map)
        self.consts_proto = self._data_type_caster(self.consts_proto,
                                                    self.data_type_cast_map)

    @classmethod
    def _data_type_caster(cls, protos, data_type_cast_map):
 
        if not data_type_cast_map:
            return protos
        result = []
        for proto in protos:
            new_proto = proto
            if proto.name in data_type_cast_map:
                new_data_type = data_type_cast_map[proto.name]
            if type(proto) == TensorProto and proto.data_type != new_data_type:
                field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
                    mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[proto.data_type]]
                vals = getattr(proto, field)
                new_proto = make_tensor(name=proto.name, data_type=new_data_type, dims=proto.dims, vals=vals)
            elif type(proto) == ValueInfoProto and proto.type.tensor_type.elem_type != new_data_type:
                new_proto.type.tensor_type.elem_type = new_data_type
            result.append(new_proto)
        return result

    def make_graph_proto(self):
        self._clean_graph()
        self._fix_data_type()

        if IS_PYTHON3:
            params = list(inspect.signature(make_graph).parameters.keys())
        else:
            params = inspect.getargspec(make_graph).args

        kwargs = {
            "initializer": self.consts_proto,
            "value_info": self.value_info_proto
        }

        return make_graph(self.nodes_proto, self._name, self.inputs_proto, self.outputs_proto, **dict([(k, kwargs[k]) for k in kwargs if k in params]))