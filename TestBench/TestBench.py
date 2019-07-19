import pprint
import numpy as np
from PIL import Image
from onnx import ModelProto, GraphProto, NodeProto
import vulkan_ep_d as vulkan_ep
##import vkFlow as vkFlow
import onnx
def o_g():
    onnx_model = onnx.load('mobilenetv2.onnx')
    graph = onnx_model.graph
    init_vals = graph.initializer
    for val, node in zip(graph.initializer, graph.node):
        dim = val.dims
        float_data = val.float_data
        layer = val.name
        print(type(node))
        print(type(val))
    vulkan_ep.create_layer()
       

     

    return layers


if (__name__ == "__main__"):
    vals = o_g()
    pp.pprint(vals)
    img = Image.open("aerial.png")
    x = np.array(img)
   # vkFlow.Run()