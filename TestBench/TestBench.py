import pprint
import numpy as np
from PIL import Image
from onnx import ModelProto, GraphProto, NodeProto
pp = pprint.PrettyPrinter(indent=4)
##import vkFlow as vkFlow
import onnx
def o_g():  
    layers = []
    onnx_model = onnx.load('mobilenetv2.onnx')
    graph = onnx_model.graph
    init_vals = graph.initializer
    for node in graph.node:
        attribute = {}
        attribute['input'] = node.input
        attribute['output'] = node.output       
        for attr in node.attribute:
            if(attr.ints != []):
                attribute[attr.name] = attr.ints
            elif(attr.floats != []):
                attribute[attr.name] = attr.floats
            elif(attr.f != 0.0):
                attribute[attr.name] = [attr.f]
        for val in graph.initializer:
            if(val.name in node.input):
                dim = val.dims
                data = {'float':val.float_data,
                        'int32':val.int32_data,
                        'int64':val.int64_data,
                        'string':val.string_data,
                        'uint64_data':val.uint64_data,
                        'double':val.double_data}
                attribute['data'] = {'dim':dim, 'data':data}
        layers.append((node.op_type, attribute))

     

    return layers


if (__name__ == "__main__"):
    vals = o_g()
    pp.pprint(vals)
    img = Image.open("aerial.png")
    x = np.array(img)
   # vkFlow.Run()