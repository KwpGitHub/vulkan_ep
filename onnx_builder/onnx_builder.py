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

type_map = {
        'STRING' :  'int',
        'STRINGS':  'Tensor*',
        'INT' :     'int',
        'INTS':     'Shape_t',
        'FLOAT':    'float',
        'FLOATS':   'Tensor*',
        'TENSOR':   'Tensor*',
        'GRAPH':    'int',
        'GRAPHS':   'int'
    }

ops = {}
op_file = open('op_file.h','w')

tmpp = '''

/*for(auto t_name : inputs){{
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
        }}*/
'''

layer_map_str = """#include <map>
#include "layers.h"
namespace backend {{

    template<typename T> Layer* createInstance(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) {{ return new T(n, i, o, a); }}

std::map<std::string, Layer*(*)(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a)> layer_map = {{
{0}
}};


std::map<std::string, std::map<std::string, std::string> > parameter_map = {{
{1}
}};


}}
    """.format



class_h_str = """#ifndef {upper}_H
#define {upper}_H //{norm}

//INPUTS:                   {input_names}
//OPTIONAL_INPUTS:          {optional_input_names}
//OUTPUS:                   {output_names}
//OPTIONAL_OUTPUTS:         {optional_output_names}
//PARAMETERS:               {parameters}
//PARAMETER_TYPES:          {parameter_types}
//OPTIONAL_PARAMETERS:      {optional_parameters}
//OPTIONAL_PARAMETERS_TYPE: {optional_parameter_types}



namespace backend {{
    class {norm} : public Layer {{
        
        vuh::Device* _get_device();

        struct Params{{
            {param_param_lst}
            //input
            {input_param_lst}
            {optional_input_param_lst}
            //output
            {output_param_lst}
            {optional_output_param_lst}
        }};

        vuh::Program<Specs, Params>* program;

    public:
        {norm}(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){{ program->run(); }}
        
        {param_lst}
        //input
        {input_lst}
        {optional_input_lst}
        //output
        {output_lst}
        {optional_output_lst}
        //std::vector<uint32_t> output_shape();
   
        ~{norm}(){{}}
    }};
}}


namespace backend {{    
    {norm}::{norm}(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {{            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\\shaders/bin/{lower}.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({{{bind_param_lst}}}, 
                            {bind_input_lst},
                            {bind_output_lst} );
    }}

    vuh::Device* {norm}::_get_device() {{
            for(auto t_name: inputs) {{
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }}
            return device;
    }}
}};

#endif
""".format_map



class_shader_str = """
#version 450
struct Shape_t {{ uint n; uint c; uint d; uint h; uint w; }};

layout(local_size_x_id = 0) in;
layout(local_size_y_id = 1) in;
layout(local_size_z_id = 2) in;

layout(push_constant) uniform Parameters {{      
    {param_param_lst}
//input
    {input_param_lst}
    {optional_input_param_lst}
//output
    {output_param_lst}
    {optional_output_param_lst}
}} params;

{shader_layout_lst}

void main(){{
    const uint idx = gl_GlobalInvocationID.x;
    const uint idy = gl_GlobalInvocationID.y;
    const uint idz = gl_GlobalInvocationID.z;
    {param_param_lst_size}
    {input_param_lst_size}
    {optional_input_param_lst_size}
    {output_param_lst_size}
    {optional_output_param_lst_size}

    if(size <= idx) {{
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
    parameter_map = list()
    layer_op_func_store = open("layer_func.json", 'w')
    layer_op = json.dump({}, layer_op_func_store)
    
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
        
        PARAMETERS =                [str(x.name) for _, x in op.attributes.items() if(x.required == True)]
        PARAMETER_TYPES =           [type_map[str(x.type).replace('AttrType.','')] for _, x in op.attributes.items() if(x.required == True)]
        OPTIONAL_PARAMETERS =       [str(x.name) for _, x in op.attributes.items() if(x.required == False)]        
        OPTIONAL_PARAMETER_TYPES =  [type_map[str(x.type).replace('AttrType.','')] for _, x in op.attributes.items() if(x.required == False)]

        INPUT_NAMES =               [str(x.name)+"_input" for x in op.inputs if(str(x.option) == 'FormalParameterOption.Single')]
        OPTIONAL_INPUT_NAMES =      [str(x.name)+"_input_o" for x in op.inputs if(str(x.option) == 'FormalParameterOption.Optional')]
        OUTPUT_NAMES =              [str(x.name)+"_output"  for x in op.outputs if(str(x.option) == 'FormalParameterOption.Single')]        
        OPTIONAL_OUTPUT_NAMES =     [str(x.name)+"_output_o" for x in op.outputs if(str(x.option) == 'FormalParameterOption.Optional')]
    
        p_map = {"inputs" :                 ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'inputs') for x in INPUT_NAMES] ),
                 "optional_input" :         ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_input')  for x in OPTIONAL_INPUT_NAMES]),
                 "outputs" :                ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'outputs')  for x in OUTPUT_NAMES]),
                 "optional_output" :        ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_output')  for x in OPTIONAL_OUTPUT_NAMES]),
                 "parameters" :             ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'parameters', y) for x,y in zip(PARAMETERS, PARAMETER_TYPES)]),
                 "optional_parameters" :    ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'optional_parameters', y)  for x,y in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES)])}

        layer_paramaters = ['{0} {1};'.format(j, i) for i, j in zip(PARAMETERS, PARAMETER_TYPES) if(j != 'Tensor*')] \
            + ['{0} {1};'.format(j, i) for i, j in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')]
        layer_parameter_tensors = [i for i, j in zip(PARAMETERS, PARAMETER_TYPES) if(j == 'Tensor*')]
        constructor_paramaters = []

        mapt = {
                'upper' :                       op_name.upper(),
                'norm' :                        op_name, 
                'lower' :                       op_name.lower(), 
                'input_names' :                 ', '.join(INPUT_NAMES),
                'optional_input_names' :        ', '.join(OPTIONAL_INPUT_NAMES),
                'output_names' :                ', '.join(OUTPUT_NAMES),
                'optional_output_names' :       ', '.join(OPTIONAL_OUTPUT_NAMES),
                'parameters' :                  ', '.join(PARAMETERS),
                'parameter_types' :             ', '.join(PARAMETER_TYPES),
                'optional_parameters' :         ', '.join(OPTIONAL_PARAMETERS),
                'optional_parameter_types' :    ', '.join(OPTIONAL_PARAMETER_TYPES),

                'param_lst' :                   ' '.join( [ '{0} {1};'.format(j, i) for i, j in zip(PARAMETERS, PARAMETER_TYPES)] + [ '{0} {1};'.format(j, i) for i, j in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES)] ) + '\n\t\t' + \
                                                ' '.join( [ 'Shape_t {0}_t;'.format(i) for i in layer_parameter_tensors]) ,
                'param_param_lst' :             ' '.join(layer_paramaters)+ '\n\t\t\t' + \
                                                ' '.join( [ 'Shape_t {0};'.format(i) for i in layer_parameter_tensors]) ,
                'param_param_lst_size' :        '\n\t'.join(['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in layer_parameter_tensors]),


                'input_lst' :                   ' '.join( ['std::string {0};'.format(x) for x in INPUT_NAMES]),
                'optional_input_lst' :          ' '.join( ['std::string {0};'.format(x) for x in OPTIONAL_INPUT_NAMES]),
                'output_lst' :                  ' '.join( ['std::string {0};'.format(x) for x in OUTPUT_NAMES]),
                'optional_output_lst' :         ' '.join( ['std::string {0};'.format(x) for x in OPTIONAL_OUTPUT_NAMES]),

                'input_param_lst' :             ' '.join( ['Shape_t {0};'.format(x) for x in INPUT_NAMES]),
                'optional_input_param_lst' :    ' '.join( ['Shape_t {0};'.format(x) for x in OPTIONAL_INPUT_NAMES]),
                'output_param_lst' :            ' '.join( ['Shape_t {0};'.format(x) for x in OUTPUT_NAMES]),
                'optional_output_param_lst' :   ' '.join( ['Shape_t {0};'.format(x) for x in OPTIONAL_OUTPUT_NAMES]),
                

                'input_param_lst_size' :             '\n\t'.join( ['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in INPUT_NAMES]),
                'optional_input_param_lst_size' :    '\n\t'.join( ['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in OPTIONAL_INPUT_NAMES]),
                'output_param_lst_size' :            '\n\t'.join( ['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in OUTPUT_NAMES]),
                'optional_output_param_lst_size' :   '\n\t'.join( ['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in OPTIONAL_OUTPUT_NAMES]),        
                


                'bind_param_lst' :              ', '.join(  ['{1}'.format(j, i) for i, j in zip(PARAMETERS, PARAMETER_TYPES) if(j != 'Tensor*')] + \
                                                            ['{1}'.format(j, i) for i, j in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')] + \
                                                            ['{0}_t'.format(i) for i in layer_parameter_tensors] + \
                                                            ['tensor_dict[{0}]->shape()'.format(i) for i in INPUT_NAMES] + \
                                                            ['tensor_dict[{0}]->shape()'.format(i) for i in OPTIONAL_INPUT_NAMES] + \
                                                            ['tensor_dict[{0}]->shape()'.format(i) for i in OUTPUT_NAMES] + \
                                                            ['tensor_dict[{0}]->shape()'.format(i) for i in OPTIONAL_OUTPUT_NAMES]  ),
                'bind_input_lst' :              ', '.join(['tensor_dict[{0}]'.format(x) for x in layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES]),
                'bind_output_lst' :             ', '.join(['tensor_dict[{0}]'.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),

                'shader_layout_lst' :           '\n'.join(['layout(std430, binding = {0}) buffer lay{0} {{ float {1}[]; }};'.format(i,x) for i,x in enumerate(layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)]),
           
                }        

        op_file.write(op.name+'=' + ', '.join(layer_paramaters) + '\n')
        
        if(op.since_version <= 10 and op.deprecated==False):
            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str(mapt))
            f.close()
          
            s_cpp = open('../_backend/shaders/' + op_name.lower() + '.comp', 'w')
            s_cpp.write(class_shader_str(mapt))
            s_cpp.close()
            layers_lst.append('#include "./layers/' + op_name.lower() + '.h"\n')
            layer_map.append('	{{ "{0}", &createInstance<{0}>}}'.format(op_name))
            parameter_map.append('{{ "{0}", {{{1}}} }}'.format(op_name, ', '.join(['{0}'.format(i) for k, i in p_map.items() if(i != '') ] )))
    
    layers.writelines(layers_lst)
    op_file.close()
    layer_map_file.write(layer_map_str(", \n".join(layer_map), ", \n".join(parameter_map)))
    layer_map_file.close()


if (__name__ == "__main__"):
    onnx_proto()
   