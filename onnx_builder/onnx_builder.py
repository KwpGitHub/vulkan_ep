import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os
import onnx_ep as onnx_ep

types = {  
            'AttrType.STRING'   :'std::string',
            'AttrType.STRINGS'  :'std::string[]',
            'AttrType.FLOAT'    :'float',
            'AttrType.FLOATS'   :'float[]',
            'AttrType.INT'      :'int',
            'AttrType.INTS'     :'int[]',
            'AttrType.TENSOR'   :'//tensor',
            'AttrType.TENSORS'  :'//std::vector<tensor>',
            'AttrType.GRAPH'    :'//graph',
            'AttrType.GRAPHS'   :'//std::vector<graph>'
        }

ops = {}
op_file = open('op_file.h','w')


def onnx_proto():
    t = onnx.defs.get_all_schemas()
    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/layers\\'))):
        os.mkdir('../_backend/layers')
    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/shaders\\'))):
        os.mkdir('../_backend/shaders')

    layers = open('../_backend/layers.h', 'w')
    layers_lst = list()
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
        lst = ['\n\t\t{} {};'.format(types[str(x.type)], x.name)  for _,x in attr.items()]
        mapt = {"upper":op_name.upper(), "norm":op_name, "lower":op_name.lower(), "param":''.join(lst)}
        class_h_str = R'''
#ifndef {upper}_H
#define {upper}_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {{
    class {norm} : public Layer {{
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {{{param}
    }};
    vuh::Program<Specs, Params>* program;

    public:
       {norm} (){{
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/{lower}.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);

        }}

        vuh::Array<float> operator(vuh::Array<float>& inpt) {{
            d_input = input;
            
            return d_output;
        }}

        void forward(){
            
        }

        {norm}& 

    }};
}}

#endif
'''.format_map(mapt) 

        class_shader_str = '''
#version 450
layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z_id = 0) in;
layout(std430, binding = 0) buffer lay0 {{ float y[]; }}; 
layout(std430, binding = 1) buffer lay1 {{ float x[]; }};
layout(push_constant) uniform Parameters {{
    uint size;
}} params;

void main() {{
    const uint id = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x
    if(id >= size) {{return;}}
}}
''' #.format_map({ "param":''.join(lst) })

        op_file.write(op.name+'=' + ', '.join(lst) + '\n')

        if(op.since_version < 8 and op.deprecated==False):
            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str)
            f.close()
            s_cpp = open('../_backend/shaders/'+op_name.lower()+'.comp', 'w')
            s_cpp.write(class_shader_str)
            s_cpp.close()

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
    onnx_proto()
    #onnx_ep.create_device()
    #onnx_ep.run()
    #onnx_model_str =  MessageToJson(onnx.load('mobilenetv2.onnx'))
    #graph = json.loads(onnx_model_str)

    #node_info = graph_def_info(graph['graph'])
    