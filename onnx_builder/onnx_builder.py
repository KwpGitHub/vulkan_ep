from collections import defaultdict 
import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os

types = {  
            'AttrType.STRING'   :'int',
            'AttrType.STRINGS'  :'int*',
            'AttrType.FLOAT'    :'float',
            'AttrType.FLOATS'   :'float*',
            'AttrType.INT'      :'int',
            'AttrType.INTS'     :'Shape_t',
            'AttrType.TENSOR'   :'Tensor*',
            'AttrType.TENSORS'  :'//std::vector<tensor>',
            'AttrType.GRAPH'    :'//graph',
            'AttrType.GRAPHS'   :'//std::vector<graph>'
        }

ops = {}
op_file = open('op_file.h','w')



layer_map_str = """#include <map>
#include "layers.h"
namespace backend {{template<typename T> Layer* createInstance(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) {{ return new T(n, i, o, a); }}

std::map<std::string, Layer*(*)(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a)> layer_map = {{
{0}
}};
}}
    """.format



class_h_str = """#ifndef {upper}_H
#define {upper}_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {{
    class {norm} : public Layer {{
        struct Params{{{param}
        }};
            {attribute_t}
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {{
            for(auto t_name: inputs) {{
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }}
            return device;
        }}

        {header_buffers}
        //parameter 
        {param}

    public:
        {norm}(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {{
            {header_inputs}
            {header_outputs}
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\\shaders/bin/{lower}.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({{{param_dict} }}, {header_inputs_2});

        }}
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){{
            {parameter_proc}   
        }}

        //Tensor* operator()(const Tensor* t) {{            
        //}}

		void forward(){{
		}}

       /* std::vector<uint32_t> output_shape(){{
            for(auto t_name : inputs){{
                if(tensor_dict.end() == tensor_dict.find(t_name) && layer_dict.end() != layer_dict.find(t_name)){{
                    //need to do math
                    return layer_dict[t_name]->output_shape();
                }}
                else if (tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){{
                    //need to do math
                    return tensor_dict[t_name]->dims;
                }}

            }}
            for(auto t_name : outputs){{
                if(tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){{
                    return tensor_dict[t_name]->dims;
                }}
            }}
        }}*/

    
        ~{norm}(){{}}

    }};
}}

#endif
""".format_map



class_shader_str = """//{norm}
#version 450
struct Shape_t {{ uint n; uint c; uint d; uint h; uint w; }};
layout(local_size_x_id = 0) in;
layout(local_size_y_id = 1) in;
layout(local_size_z_id = 2) in;  

layout(push_constant) uniform Parameters {{      
    {param}
}} params;

{shader_buffers}

void main(){{
    const uint id = gl_GlobalInvocationID.x; 
    const uint size = params.input.n * params.input.c * params.input.d * params.input.h * params.input.w;
    if(size <= id) {{
        return;
    }}
   
}}
""".format_map



def onnx_proto():
    t = onnx.defs.get_all_schemas()
    ls = onnx.defs.get_function_ops()
    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/layers\\'))):
        os.mkdir('../_backend/layers')
    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/shaders\\'))):
        os.mkdir('../_backend/shaders')
    
    layers = open('../_backend/layers.h', 'w')
    layer_map_file = open("../_backend/layers_map.h", 'w')
    layers_lst = list()
    layer_map = list()
    layer_op_func_store = open("layer_func.json", 'w')
    layer_op = json.dump({}, layer_op_func_store)
    
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
        shader_buffers  =   ["layout(std430, binding = {0}) buffer lay{0} {{ float {1}[]; }};\n".format(i, x.name) for i,x in enumerate(op.outputs + op.inputs)]
        
        attributes_t    =   ['\n\t\t{} {};'.format('Tensor*', x.name) for _,x in attr.items() if(str(x.type) == 'AttrType.FLOATS' or str(x.type) == 'AttrType.TENSOR')]       
        parameters      =   ['Shape_t {}_t;'.format(x.name) for x in (op.inputs + op.outputs)] +  ['{} {}_t;'.format(types[str(x.type)], x.name)  for _,x in attr.items() if(str(x.type) != 'AttrType.FLOATS' and str(x.type) != 'AttrType.TENSOR')]
        parameter_dict  =   ['{}_t'.format(x.name) for x in (op.inputs + op.outputs)] + ['{}_t'.format(x.name)  for _,x in attr.items() if(str(x.type) != 'AttrType.FLOATS' and str(x.type) != 'AttrType.TENSOR')]
        parameter_proc  =   ['convert_vec_param(a["{0}"], {0}_t);'.format(x.name) for x in (op.inputs + op.outputs)] + ['convert_vec_param(a["{0}"], {0}_t);'.format(x.name)  for _,x in attr.items() if(str(x.type) != 'AttrType.FLOATS' and str(x.type) != 'AttrType.TENSOR')]
        mapt = {
                'upper'             :   op_name.upper(),
                'norm'              :   op_name, 
                'lower'             :   op_name.lower(), 
                'param'             :   ' '.join(parameters) , 
                'param_dict'        :   ', '.join(parameter_dict),
                'parameter_proc'    :   '\n\t\t\t'.join(parameter_proc),
                'attribute_t'       :   ''.join(attributes_t), 
                'shader_buffers'    :   ''.join(shader_buffers),
                'header_buffers'    :   ' '.join(["std::string {0};".format(x.name) for i, x in enumerate(op.inputs + op.outputs)]),
                'header_inputs'     :   ' '.join(["{0} = i[{1}];".format(x.name, i) for i, x in enumerate(op.inputs)]),
                'header_outputs'    :   ' '.join(["{0} = o[{1}];".format(x.name, i) for i, x in enumerate(op.outputs)]),
                'header_inputs_2'   :   ', '.join(["tensor_dict[{0}]".format(x.name) for i, x in enumerate(op.outputs + op.inputs)]),
        }
        

        op_file.write(op.name+'=' + ', '.join(parameters) + '\n')
        



        if(op.since_version <= 10 and op.deprecated==False):
            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str(mapt))
            f.close()
            s_cpp = open('../_backend/shaders/' + op_name.lower() + '.comp', 'w')
            s_cpp.write(class_shader_str(mapt))
            s_cpp.close()
            layers_lst.append('#include "./layers/' + op_name.lower() + '.h"\n')
            layer_map.append('	{{ "{0}", &createInstance<{0}>}}'.format(op_name))

    
    layers.writelines(layers_lst)
    op_file.close()
    layer_map_file.write(layer_map_str(",\n".join(layer_map)))
    layer_map_file.close()


if (__name__ == "__main__"):
    onnx_proto()
   