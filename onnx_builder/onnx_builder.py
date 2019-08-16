from collections import defaultdict 
import numpy as np
from google.protobuf.json_format import MessageToJson
import onnx
import json
import os

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

layer_map_str = """#include <map>
#include "layers.h"
namespace backend {{

    template<typename T> Layer* createInstance(std::string n) {{ return new T(n); }}

std::map<std::string, Layer*(*)(std::string n)> layer_map = {{
/*
{0}
*/
}};


std::map<std::string, std::map<std::string, std::string> > parameter_map = {{
/*
{1}
*/
}};


}}
    """.format



class_h_str = """#include "../layer.h"
#ifndef {upper}_H
#define {upper}_H 
/*
{doc}
//*/
//{norm}
//INPUTS:                   {input_names}
//OPTIONAL_INPUTS:          {optional_input_names}
//OUTPUS:                   {output_names}
//OPTIONAL_OUTPUTS:         {optional_output_names}
//PARAMETERS:               {parameters}
//PARAMETER_TYPES:          {parameter_types}
//OPTIONAL_PARAMETERS:      {optional_parameters}
//OPTIONAL_PARAMETERS_TYPE: {optional_parameter_types}

//class stuff
namespace backend {{   

    class {norm} : public Layer {{
        typedef struct {{
            {param_param_lst}
            {input_param_lst}
            {optional_input_param_lst}
            {output_param_lst}
            {optional_output_param_lst}
        }} binding_descriptor;

        {param_lst}
        {input_lst}
        {optional_input_lst}
        {output_lst}
        {optional_output_lst}

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        {norm}(std::string n);
    
        void forward() {{ program->run(); }}
        
        void init({init_param_lst}); 
        void bind({bind_lst}); 

        ~{norm}() {{}}

    }};
    
}}

#endif

""".format_map

cpp_class_str = """#include "{norm}.h"

//cpp stuff
namespace backend {{    
   
    {norm}::{norm}(std::string n) : Layer(n) {{ }}
       
    vuh::Device* {norm}::_get_device() {{
        
        return device;
    }}
    
    void {norm}::init({init_param_lst}) {{      
{init_input_lst_3}  
    }}
    
    void {norm}::bind({bind_lst}){{
        {bind_input_lst_2}
{bind_input_lst_1}
{bind_output_lst_1}
{bind_binding_lst_1}
{bind_binding_tensor_1}
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/{lower}.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding{bind_param_lst}{bind_input_lst}{bind_output_lst});
    }}
    
}}

    //backend::nn;

//python stuff


""".format_map

python_class_str = """
#from _backend.nn import {norm} as c_{norm}
class {norm}:
    name = None
{python_variables}
    #parameters
{python_parameters}
    def __init__(self, name):
        self.name = name

    def input(self, *args):
        inpts = [{python_inputs_func}]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = [{python_outputs_func}]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['{lower}'] = {norm}

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
    activation = [ 'Tanh', 'Acos', 'Asin', 'Atan', 'Cos', 'Sin', 'Tan', 'Sinh', 'Cosh', 'Asinh', 'Acosh', 'Atanh', 'Softplus', 'Softsign', 'Sigmoid', 'Relu', 'PRelu', 'Elu', 'HardSigmoid', 'Hardmax', 'Selu', 'LogSoftmax', 'Softmax']
    elementwise = ['Abs', 'Neg', 'Exp', 'Ceil', 'Not', 'Floor', 'Log', 'IsNaN', 'Sqrt', 'Sign', 'Erf', 'NonZero']
    math_op = ['Add', 'And', 'Mul', 'Div', 'Sub', 'Or', 'Pow', 'Xor', 'Min', 'Max', 'Sum']
    
    single_input = []
    single_output = []
    single_element = []
    double_element = []
    complex_element = []
    simple_element = []


    layers = open('../_backend/layers.h', 'w')
    layer_map_file = open("../_backend/layers_map.h", 'w')
    py_layers = open('../TestingPipeline/layers.py', 'w')
    layers_lst = list()
    layer_map = list()
    parameter_map = list()
    py_layers_map = list()
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
        OPTIONAL_INPUT_NAMES =      [str(x.name)+"_input_opt" for x in op.inputs if(str(x.option) == 'FormalParameterOption.Optional')]
        OUTPUT_NAMES =              [str(x.name)+"_output"  for x in op.outputs if(str(x.option) == 'FormalParameterOption.Single')]        
        OPTIONAL_OUTPUT_NAMES =     [str(x.name)+"_output_opt" for x in op.outputs if(str(x.option) == 'FormalParameterOption.Optional')]
    
        p_map = {"inputs" :                 ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'inputs') for x in INPUT_NAMES] ),
                 "optional_input" :         ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_input')  for x in OPTIONAL_INPUT_NAMES]),
                 "outputs" :                ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'outputs')  for x in OUTPUT_NAMES]),
                 "optional_output" :        ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_output')  for x in OPTIONAL_OUTPUT_NAMES]),
                 "parameters" :             ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'parameters', y) for x,y in zip(PARAMETERS, PARAMETER_TYPES)]),
                 "optional_parameters" :    ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'optional_parameters', y)  for x,y in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES)])}

        layer_paramaters = ['{0} {1};'.format(j, i) for i, j in zip(PARAMETERS, PARAMETER_TYPES) if(j != 'Tensor*')] \
            + ['{0} {1};'.format(j, i) for i, j in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')]
        layer_parameter_tensors = [i for i, j in zip(PARAMETERS, PARAMETER_TYPES) if(j == 'Tensor*')] + [i for i, j in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES) if(j == 'Tensor*')]

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
                'doc':                          op.doc.replace('/*', '//').replace('*/', '//') + '\n' + '\n'.join(['input: ' + x.description for x in op.inputs]).replace('/*', '//').replace('*/', '//') + '\n' + '\n'.join(['output: ' + x.description for x in op.outputs]).replace('/*', '//').replace('*/', '//'),
                'param_lst' :                   ' '.join( [ '{0} {1};'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')] + [ 'std::string {1};'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j == 'Tensor*')] ),

                # 'param_lst_lst' :               ' '.join( [ '{0}={0};'.format(i) for i in PARAMETERS + OPTIONAL_PARAMETERS]),
                'param_param_lst' :             ' '.join(layer_paramaters)+ '\n\t\t\t' + \
                                                ' '.join([ 'Shape_t {0};'.format(i) for i in layer_parameter_tensors]) ,
                'param_param_lst_size' :        '\n\t'.join(['const uint {0}_size = params.{0}.n * params.{0}.c * params.{0}.d * params.{0}.w * params.{0}.h;'.format(x) for x in layer_parameter_tensors]),
                'constructor_param_type' :      ''.join( [ ', {0}'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if (j != 'Tensor*')]),

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
                                 
                'init_param_lst' :       ', '.join( [ ' {0} _{1}'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if (j != 'Tensor*')]),
                'init_input_lst_3':             ' '.join(['\t\t {0} = _{0}; \n'.format(i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')]),
             
                'bind_lst' :                    ', '.join(['std::string _{0}'.format(x) for x in layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),              
                'bind_input_lst_2':             ' '.join(['{0} = _{0};'.format(x) for x in layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'bind_input_lst' :              ''.join([', *tensor_dict[{0}]->data()'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES ]),
                'bind_output_lst' :             ''.join([', *tensor_dict[{0}]->data()'.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'bind_param_lst' :              ''.join([', *tensor_dict[{0}]->data()'.format(x) for x in layer_parameter_tensors]),
                'bind_input_lst_1' :              ' '.join(['\t\tbinding.{0} = tensor_dict[{0}]->shape();\n '.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES]),
                'bind_output_lst_1' :             ' '.join(['\t\tbinding.{0} = tensor_dict[{0}]->shape();\n '.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'bind_binding_lst_1' :            ' '.join(['\t\tbinding.{0} = {0};\n '.format(i)for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*')]),
                'bind_binding_tensor_1' :         ' '.join(['\t\tbinding.{0} = tensor_dict[{0}]->shape();\n '.format(i)for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j == 'Tensor*')]),

                'shader_layout_lst' :           '\n'.join(['layout(std430, binding = {0}) buffer lay{0} {{ float {1}[]; }};'.format(i,x) for i,x in enumerate(layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)]),
                
                'call_python_binding' :         ', '.join(['std::string' for _ in layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                
                'pybind11_constructor':         ''.join([ ', {0}'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if (j != 'Tensor*')]).replace("Shape_t", "backend::Shape_t"),
                'python_variables' :            ''.join(['    {0} = None\n'.format(x) for x in layer_parameter_tensors + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'python_parameters' :           ''.join(['    {0} = None\n'.format(i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES) if(j != 'Tensor*') ]),
                'python_inputs_func' :          ', '.join('"{0}"'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES),
                'python_outputs_func' :         ', '.join('"{0}"'.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)
                }        

        py_layers_map.append(python_class_str(mapt))
       

        op_file.write(op.name+'=' + ', '.join(layer_paramaters) + '\n')
        
        if(op.since_version <= 10 and op.deprecated==False):
            if(len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) == 1 and len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) == 1 and len(PARAMETERS + OPTIONAL_PARAMETERS) == 0 and op_name not in activation and op_name not in elementwise and op_name not in math_op):
                single_element.append(op_name)
            elif(len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) == 2 and len(PARAMETERS + OPTIONAL_PARAMETERS) == 0 and op_name not in activation and op_name not in elementwise and op_name not in math_op):
                double_element.append(op_name)
            elif(len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) == 1 and  len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) == 0 and op_name not in activation and op_name not in elementwise and op_name not in math_op):
                single_input.append(op_name)
            elif(len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) == 0 and  len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) == 1 and op_name not in activation and op_name not in elementwise and op_name not in math_op):
                single_output.append(op_name)
            elif((len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) == 1 or len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) == 1) and  len(PARAMETERS + OPTIONAL_PARAMETERS) <= 3 and op_name not in activation and op_name not in elementwise and op_name not in math_op):
                simple_element.append(op_name)        
            elif(op_name not in activation and op_name not in elementwise and op_name not in math_op):
                complex_element.append(op_name)


            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str(mapt))
            f.close()
            f_cpp = open("../_backend/layers/" + op_name.lower() + '.cpp', 'w')
            f_cpp.write(cpp_class_str(mapt))
            f_cpp.close()
            s_cpp = open('../_backend/shaders/' + op_name.lower() + '.comp', 'w')
            s_cpp.write(class_shader_str(mapt))
            s_cpp.close()
            layers_lst.append('#include "./layers/' + op_name.lower() + '.h"\n')
            layer_map.append('	{{ "{0}", &createInstance<{0}>}}'.format(op_name))
            parameter_map.append('{{ "{0}", {{{1}}} }}'.format(op_name, ', '.join(['{0}'.format(i) for k, i in p_map.items() if(i != '') ] )))

    
    layers.writelines(layers_lst)
    op_file.close()
    layer_map_file.write(layer_map_str(", \n".join(layer_map), ", \n".join(parameter_map)))
    py_layers.write('layer_map = {}' + '\n\n'.join(py_layers_map))
    print(single_element)
    print(double_element)
    print(single_input)
    print(single_output)
    print(simple_element)
    print(complex_element)

    layer_map_file.close()


if (__name__ == "__main__"):
    onnx_proto()
   