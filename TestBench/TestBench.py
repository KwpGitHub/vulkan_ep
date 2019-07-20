import numpy as np
from PIL import Image
from google.protobuf.json_format import MessageToJson
#import onnx_ep as onnx_ep
import onnx
import json

def node_info(graph, i):
    return (graph['node'][i], graph['initializer'][i])

if (__name__ == "__main__"):
    onnx_model = onnx.load('mobilenetv2.onnx')
    
    dict_str = MessageToJson(onnx_model, preserving_proto_field_name=True)
    graph_dict = json.loads(dict_str)
    graph_size = len(graph_dict['graph']['node'])
    graph = graph_dict['graph']

    print(graph_size)

    for i in range(3,graph_size):
        node, data = node_info(graph ,i)
        print(node)
        print('\n\n')
        print(data)
        print("\n\n")
        break
#        onnx_ep.create_layer(node, data)

    img = Image.open("aerial.png")
    x = np.array(img)
    