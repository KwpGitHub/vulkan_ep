import numpy as np
from google.protobuf.json_format import MessageToJson
#import onnx_ep as onnx_ep
import onnx
import json

ops = {}

def node_info(graph, i):
    return (graph['node'][i], graph['initializer'][i])

if (__name__ == "__main__"):
    onnx_model = onnx.load('mobilenetv2.onnx')
    t = onnx.defs.get_all_schemas()
    for op in t:
        ops[op.name] = op
    print(len(ops.keys()))
    x = MessageToJson(t)
    dict_ = json.loads(x)
    