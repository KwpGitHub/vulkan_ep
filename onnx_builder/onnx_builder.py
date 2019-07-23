import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_ep as onnx_ep

types = {   'AttrType.STRING'   :'std::string',
            'AttrType.STRINGS'  :'std::vector<string>',
            'AttrType.FLOAT'    :'float',
            'AttrType.FLOATS'   :'std::vector<float>',
            'AttrType.INT'      :'int',
            'AttrType.INTS'     :'std::vector<int>',
            'AttrType.TENSOR'   :'tensor',
            'AttrType.TENSORS'  :'std::vector<tensor>',
            'AttrType.GRAPH'    :'graph',
            'AttrType.GRAPHS'   :'std::vector<graph>'
        }

ops = {}
op_file = open('op_file.h','w')

class_h = R"""#include <vector>
namespace backend {
    class %s {
    public:
        %s ();
        ~%s();
    private:
%s
    };
}
"""

class_cpp = R"""#include "%s.h"

namespace backend {
    %s::%s() {
        
    }

    ~%s::%s() {
        
    }
}
"""

def onnx_proto():
    t = onnx.defs.get_all_schemas()
    if(not os.path.isdir(os.path.join(os.getcwd(),'layers\\'))):
        os.mkdir('layers')
    layers = open('layers.h', 'w')
    layers_lst = list()
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
        lst = ['\n\t\t{} {};'.format(types[str(x.type)], x.name)  for _,x in attr.items()]

        class_h_str = class_h % (op.name, op.name, op.name, ''.join(lst))
        class_cpp_str = class_cpp % (op_name.lower(), op.name, op.name, op.name, op.name)
        op_file.write(op.name+'=' + ', '.join(lst) + '\n')

        if(op.since_version < 8 and op.deprecated==False):
            f_cpp = open('./layers/'+op_name.lower()+'.cpp', 'w')        
            f_cpp.write(class_cpp_str)
            f_cpp.close()
            f = open('./layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str)
            f.close()
            layers_lst.append('#include "./layers/'+op_name.lower()+'.h"\n')
    layers.writelines(layers_lst)
    op_file.close()

def graph_def_info(graph):
    nodes = {}
    init_vals = {}
    
    for data in graph['initializer']:
        init_vals[data['name']] = data

    for node in graph['node']:
        nodes[node['name']] = node
        for i, input in enumerate(node['input']):
            for data_names in init_vals.keys():
                if(data_names == input):
                    nodes[node['name']]['input'][i] = init_vals[data_names]
    return nodes

    


if (__name__ == "__main__"):
    #onnx_proto()
    onnx_ep.create_device()
    onnx_ep.run()
    #onnx_model_str =  MessageToJson(onnx.load('mobilenetv2.onnx'))
    #graph = json.loads(onnx_model_str)

    #node_info = graph_def_info(graph['graph'])
    