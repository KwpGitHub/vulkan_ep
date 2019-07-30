import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os

types = {  
            'AttrType.STRING'   :'std::string',
            'AttrType.STRINGS'  :'//std::string[]',
            'AttrType.FLOAT'    :'float',
            'AttrType.FLOATS'   :'//float[]',
            'AttrType.INT'      :'int',
            'AttrType.INTS'     :'//int[]',
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
    layer_map_file = open("../_backend/layers_map.h", 'w')
    layers_lst = list()
    layer_map = list()
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
        lst = ['\n\t\t{} {};'.format(types[str(x.type)], x.name)  for _,x in attr.items()]
        lst_t = [', {} {}'.format(types[str(x.type)], x.name)  for _,x in attr.items()]

        mapt = {"upper":op_name.upper(), "norm":op_name, "lower":op_name.lower(), "param":''.join(lst) , 'param_t':''.join(lst_t)}
        class_h_str = R'''#ifndef {upper}_H
#define {upper}_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {{
    class {norm} : public Layer {{
    public:
        {norm}(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {{}}
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {{
            
        }}

        void forward(){{
        
        }}

        void build_pipeline(){{
            
        }}

        ~{norm}(){{}}

    }};
}}

#endif
'''.format_map(mapt) 

        class_shader_str = '''//{name}
#version 450

layout(local_size_x_id = 0) in;
layout(local_size_y_id = 1) in;
layout(local_size_z_id = 2) in;  

layout(push_constant) uniform Parameters {{
    uint size;
}} params;

layout(std430, binding = 1) buffer lay1 {{ float x[]; }};

void main(){{
    const uint id = gl_GlobalInvocationID.x; 
    if(params.size <= id){{
        return;
    }}
    x[id] = x[id];
}}
'''.format_map({ "name":op_name, "param":''.join(lst) })

        op_file.write(op.name+'=' + ', '.join(lst) + '\n')

        if(op.since_version < 8 and op.deprecated==False):
            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str)
            f.close()
            s_cpp = open('../_backend/shaders/'+op_name.lower()+'.comp', 'w')
            s_cpp.write(class_shader_str)
            s_cpp.close()
            layers_lst.append('#include "./layers/'+op_name.lower()+'.h"\n')
            layer_map.append('	{{ "{0}", &{0} }}'.format(op_name))

    
    layers.writelines(layers_lst)
    op_file.close()
    layer_map_str = """#include <map>
#include "layers.h"
namespace backend {{
std::map<const char*, Layer> layer_map = {{
{0}
}};
}}
    """.format(",\n".join(layer_map))
    layer_map_file.write(layer_map_str)
    layer_map_file.close()


if (__name__ == "__main__"):
    onnx_proto()
   