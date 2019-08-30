from collections import defaultdict 
import numpy as np
from google.protobuf.json_format import MessageToJson
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case.node import expect

import onnx
import json
import os

type_map = {
    'STRING' :  'std::string',
    'STRINGS':  'std::vector<std::string>',
    'INT' :     'int',
    'INTS':     'std::vector<int>',
    'FLOAT':    'float',
    'FLOATS':   'std::vector<float>',
    'TENSOR':   'std::vector<float>',
    'GRAPH':    'int',
    'GRAPHS':   'int'
}

ops = {}
op_file = open('op_file.h','w')

layer_map_str = """/*#include <map>
#include "layers.h"
namespace backend {{


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

}}*/
    """.format


class_h_str = """#ifndef {upper}_H
#define {upper}_H 

#include "../layer.h"

/*
{doc}
*/

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
namespace layers {{   

    class {norm} : public backend::Layer {{
        typedef struct {{
            uint32_t size;
        }} binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        {param_lst}
        {input_lst}
        {optional_input_lst}
        {output_lst}
        {optional_output_lst}

        binding_descriptor   binding;
       

    public:
        {norm}(std::string name);
        
        virtual void forward();        
        virtual void init({init_param_lst}); 
        virtual void bind({bind_lst}); 
        virtual void build();

        ~{norm}() {{}}
    }};
   
}}
#endif

""".format_map


cpp_class_str = """#include "{lower}.h"
//cpp stuff
namespace layers {{    
   
    {norm}::{norm}(std::string name) : backend::Layer(name) {{    
        file.append(backend::file_path);
        file.append("shaders/bin/{lower}.spv");       
        dev = backend::g_device;
    }}
       
        
    void {norm}::init({init_param_lst}) {{      
{init_input_lst_3}  

    }}
    
    void {norm}::bind({bind_lst}){{    
        {bind_input_lst_2}        
{bind_input_lst_1}
{bind_output_lst_1}
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }}

    void {norm}::build(){{     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({{128}}, *_SHAPES{bind_input_lst}{bind_output_lst});
    }}

    void {norm}::forward(){{ 
        program->run();
    }}

}}

""".format_map


class_shader_str = """
#version 450
struct Shape_t {{ uint n; uint c; uint d; uint h; uint w; }};

layout(local_size_x_id = 0) in;
layout(local_size_y_id = 1) in;
layout(local_size_z_id = 2) in;

layout(push_constant) uniform Parameters {{      
   uint x;
}} params;

layout(std430, binding = 0) buffer lay0 {{ Shape_t shape[]; }};
{shader_layout_lst}


void main(){{
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint z = gl_GlobalInvocationID.z;

{shader_shape_lst}
    uint n = shape[0].n;
    
    if(x >= shape[0].w || y >= shape[0].h || z >= shape[0].d){{
        return;
    }}
    for(uint i = 0; i < n; i++){{
        for(uint j = 0; j < shape[0].c; j++){{
           

            uint indx = x + uint(y*{shader_input_shape}.x)\
                        + uint(z*{shader_input_shape}.x*{shader_input_shape}.y)\
                        + uint(j*{shader_input_shape}.x*{shader_input_shape}.y*{shader_input_shape}.z)\
                        + uint(i*{shader_input_shape}.x*{shader_input_shape}.y*{shader_input_shape}.z*{shader_input_shape}.w);

            {shader_input}[indx] = {shader_output}[indx];
        }}
    }}
       

}}
""".format_map


layers_file_str = '''
void init_layer_{norm}(py::module& m){{
    m.def("_{norm}", [](py::str name{create_param_lst}) {{
        layers::{norm}* layer = new layers::{norm}(std::string(name));
        layer->init({create_init_lst});
        layer->bind({create_bind_lst});
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "{norm}" <<std::endl;

    }});

    m.def("_{norm}_run",  [](py::str name) {{
        //std::cout << "RUN ::: " << std::string(name) << " ::: {norm}" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    }});

}}\n\n'''.format_map


python_class_str = """

class {norm}:
    name = None
{python_variables}
    #parameters
{python_parameters}
    input_params = [{python_inputs_func}]
    output_params = [{python_outputs_func}]
    attribute_params = [{python_attribute_func}]
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._{norm}
        self.run_ = nn._{norm}_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
        return self

    def build(self):
        self.Module(self.name, {python_call})

    def run(self):
        self.run_(self.name)

    def test(self):
{python_testing_code}

layer_map['{norm}'] = {norm}

""".format_map


def onnx_proto():
    t = onnx.defs.get_all_schemas()
    ls = onnx.defs.get_function_ops()
    snippets = collect_snippets()
    sample_implementation = collect_sample_implementations()

    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/layers\\'))):
        os.mkdir('../_backend/layers')
    if(not os.path.isdir(os.path.join(os.getcwd(),'../_backend/shaders\\'))):
        os.mkdir('../_backend/shaders')
    
    
        
    activation = [ 'Tanh', 'Acos', 'Asin', 'Atan', 'Cos', 'Sin', 'Tan', 'Sinh', 'Cosh', 'Asinh', 'Acosh', 'Atanh', 'Softplus', 'Softsign', 'Sigmoid', 'Relu', 'PRelu', 'Elu', 'HardSigmoid', 'Hardmax', 'Selu', 'LogSoftmax', 'Softmax']
    elementwise = ['Abs', 'Neg', 'Exp', 'Ceil', 'Not', 'Floor', 'Log', 'IsNaN', 'Sqrt', 'Sign', 'Erf', 'NonZero']
    math_op = ['Add', 'And', 'Mul', 'Div', 'Sub', 'Or', 'Pow', 'Xor', 'Min', 'Max', 'Sum']
    
    dump = list()
    single_input = []
    single_output = []
    single_element = []
    double_element = []
    complex_element = []
    simple_element = []
    
    layers = open('../_backend/layers.hpp', 'w')
    layer_map_file = open("../_backend/layers_map.h", 'w')
    py_layers = open('../TestingPipeline/layers.py', 'w')
    pybind_modules_file = open('../_backend/pybind_modules.txt', 'w')
    pybind_modules = list()
    layers_lst = list()
    layer_map = list()
    parameter_map = list()
    py_layers_map = list()
    
    for op in t:
        ops[op.name] = op
        op_name = str(op.name)
        attr = op.attributes
            
        PARAMETERS =                [str(x.name) for _, x in op.attributes.items() if(x.required == True)]
        PARAMETER_TYPES =           [type_map[str(x.type).replace('AttrType.','')] for _, x in op.attributes.items() if(x.required == True)]
        OPTIONAL_PARAMETERS =       [str(x.name) for _, x in op.attributes.items() if(x.required == False)]        
        OPTIONAL_PARAMETER_TYPES =  [type_map[str(x.type).replace('AttrType.','')] for _, x in op.attributes.items() if(x.required == False)]

        INPUT_NAMES =               [str(x.name)+"_i" for x in op.inputs if(str(x.option) == 'FormalParameterOption.Single')]
        OPTIONAL_INPUT_NAMES =      [str(x.name)+"_i" for x in op.inputs if(str(x.option) == 'FormalParameterOption.Optional')]
        OUTPUT_NAMES =              [str(x.name)+"_o"  for x in op.outputs if(str(x.option) == 'FormalParameterOption.Single')]        
        OPTIONAL_OUTPUT_NAMES =     [str(x.name)+"_o" for x in op.outputs if(str(x.option) == 'FormalParameterOption.Optional')]
        
        code = list()
        sample = 'pass'
        if(op_name in snippets):
            for x in snippets[op_name]:
                code.append(x)
        _code = ''.join("\n        def _{0}():\n            {1}".format(n.upper(), c.replace('\n', '\n            ')) for n, c in code) if(len(code) != 0) else '        pass'

        if(op_name.lower() in sample_implementation):
            sample = sample_implementation[op_name.lower()]
        if(op_name in ['Concat', 'Sum', 'Scan', 'Mean']): 
            OPTIONAL_INPUT_NAMES = ['x'+str(i)+'_i' for i in range(32)]
        if(op_name in ['Scan']):
            OPTIONAL_OUTPUT_NAMES = ['y'+str(i)+'_o' for i in range(32)]
        dump.append({
                'op' : op_name,
                'input' : (INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0] if (len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) != 0) else (OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0],
                'output' : (OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0] if (len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) != 0) else (INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0],
                'python_op' : '',
                'shader_op' : '',
                'kernel_op' : '',
        })


        p_map = {
                "inputs" :                 ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'inputs') for x in INPUT_NAMES] ),
                 "optional_input" :         ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_input')  for x in OPTIONAL_INPUT_NAMES]),
                 "outputs" :                ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'outputs')  for x in OUTPUT_NAMES]),
                 "optional_output" :        ", ".join(['{{"{0}", {{"{1}", "Tensor*"}} }}'.format(x, 'optional_output')  for x in OPTIONAL_OUTPUT_NAMES]),
                 "parameters" :             ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'parameters', y) for x,y in zip(PARAMETERS, PARAMETER_TYPES)]),
                 "optional_parameters" :    ", ".join(['{{"{0}", {{"{1}", "{2}"}} }}'.format(x, 'optional_parameters', y)  for x,y in zip(OPTIONAL_PARAMETERS, OPTIONAL_PARAMETER_TYPES)])
                 }

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
                'param_lst' :                   ' '.join( [ '{0} m_{1};'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]),

               
                'constructor_param_type' :      ''.join( [ ', {0}'.format(i) for i in PARAMETERS + OPTIONAL_PARAMETERS]),

                'input_lst' :                   ' '.join( ['std::string m_{0};'.format(x) for x in INPUT_NAMES]),
                'optional_input_lst' :          ' '.join( ['std::string m_{0};'.format(x) for x in OPTIONAL_INPUT_NAMES]),
                'output_lst' :                  ' '.join( ['std::string m_{0};'.format(x) for x in OUTPUT_NAMES]),
                'optional_output_lst' :         ' '.join( ['std::string m_{0};'.format(x) for x in OPTIONAL_OUTPUT_NAMES]),
                                               
                'init_param_lst' :              ', '.join( [ ' {0} _{1}'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]),
                'init_input_lst_3':             ' '.join(['\t\t m_{0} = _{0}; \n'.format(i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]),
                
                'bind_lst' :                    ', '.join(['std::string _{0}'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),              
                'bind_input_lst_2':             ' '.join(['m_{0} = _{0};'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'bind_input_lst' :              ''.join([', *backend::tensor_dict[m_{0}]->data'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES ]),
                'bind_output_lst' :             ''.join([', *backend::tensor_dict[m_{0}]->data'.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                
                'bind_input_lst_1' :              ' '.join(['\t\tSHAPES.push_back(backend::tensor_dict[m_{0}]->shape());\n '.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES]),
                'bind_output_lst_1' :             ' '.join(['\t\tSHAPES.push_back(backend::tensor_dict[m_{0}]->shape());\n '.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'bind_binding_lst_1' :            ' '.join(['\t\t//binding.{0} = {0};\n '.format(i)for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]),

                'shader_layout_lst' :           '\n'.join(['layout(std430, binding = {0}) buffer lay{0} {{ float {1}[]; }};'.format(i,x) for i,x in enumerate(INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES, start=1)]),
                'shader_shape_lst' :            '\n'.join(['\tvec4 {0}_shape = vec4(shape[{1}].c, shape[{1}].d, shape[{1}].h, shape[{1}].w);'.format(j,i) for i, j in enumerate(INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)]),
                'shader_input' :                '{0}'.format((INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0] if (len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) != 0) else (OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0]),
                'shader_output' :               '{0}'.format((OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0] if (len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) != 0) else (INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0]),
                'shader_input_shape' :          '{0}_shape'.format((INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0] if (len(INPUT_NAMES + OPTIONAL_INPUT_NAMES) != 0) else (OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0]),
                'shader_output_shape' :         '{0}'.format((OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES)[0] if (len(OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES) != 0) else (INPUT_NAMES + OPTIONAL_INPUT_NAMES)[0]),

                'call_python_binding' :         ', '.join(['std::string' for _ in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                
                'pybind11_constructor':         ''.join([ ', {0}'.format(j, i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]).replace("Shape_t", "backend::Shape_t"),
                'python_variables' :            ''.join(['    {0} = str()\n'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                'python_parameters' :           ''.join(['    {0} = {1}()\n'.format(i,j) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]).replace('std::vector<std::string>', 'list').replace('std::string', 'str').replace('std::vector<float>', 'list').replace('std::vector<int>', 'list'),
                'python_inputs_func' :          ', '.join('"{0}"'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES),
                'python_outputs_func' :         ', '.join('"{0}"'.format(x) for x in OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES),
                'python_attribute_func' :       ', '.join('"{0}"'.format(x) for x in PARAMETERS + OPTIONAL_PARAMETERS),
                'python_call' :                 ', '.join('self.{0}'.format(x) for x in PARAMETERS + OPTIONAL_PARAMETERS + INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES),
                'python_testing_code' :         _code + ''.join('\n        _{0}()'.format(n.upper()) for n,_ in code) ,
                'create_param_lst' :            ' '.join([', {0} _{1}'.format(j,i) for i, j in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)] + [ ', std::string _{0}'.format(i) for i in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]).replace('std::vector<std::string>', 'py::list').replace('std::string', 'py::str').replace('std::vector<float>', 'py::list').replace('std::vector<int>', 'py::list'),
                'create_init_lst' :             ', '.join(['_{0}'.format(x) if('std::vector' not in i) else 'backend::convert<{1}>(_{0})'.format(x, i .replace('std::vector<','').replace('>','')) for x, i in zip(PARAMETERS + OPTIONAL_PARAMETERS, PARAMETER_TYPES + OPTIONAL_PARAMETER_TYPES)]),
                'create_bind_lst' :             ', '.join(['_{0}'.format(x) for x in INPUT_NAMES + OPTIONAL_INPUT_NAMES + OUTPUT_NAMES + OPTIONAL_OUTPUT_NAMES]),
                
        }        

        op_file.write(op.name+'=' + ', '.join(PARAMETERS + OPTIONAL_PARAMETERS) + '\n')
        
        if(op.since_version <= 10 and op.deprecated==False):
            py_layers_map.append(python_class_str(mapt))
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

            pybind_modules.append('init_layer_{norm}(nn);'.format_map(mapt) )

            f = open('../_backend/layers/'+op_name.lower()+'.h', 'w')
            f.write(class_h_str(mapt))
            f.close()
            f_cpp = open("../_backend/layers/" + op_name.lower() + '.cpp', 'w')
            f_cpp.write(cpp_class_str(mapt))
            f_cpp.close()
            s_cpp = open('../_backend/shaders/' + op_name.lower() + '.comp', 'w')
            s_cpp.write(class_shader_str(mapt))
            s_cpp.close()
            layers_lst.append( 'void init_layer_{norm}(py::module&);\n#include "./layers/{lower}.h"'.format_map(mapt) + layers_file_str(mapt) )
            layer_map.append('	{{ "{0}", &createInstance<{0}>}}'.format(op_name))
            parameter_map.append('{{ "{0}", {{{1}}} }}'.format(op_name, ', '.join(['{0}'.format(i) for k, i in p_map.items() if(i != '') ] )))
            
    layer_map_file.write(layer_map_str(", \n".join(layer_map), ", \n".join(parameter_map)))
   
    #layer_op_func_store = open("layer_func.json", 'w')
    #layer_op = json.dump(dump, layer_op_func_store)


    pybind_modules_file.write('\n'.join(pybind_modules))
    layers.writelines(layers_lst)
    op_file.close()

    layer_map_file.write(layer_map_str(", \n".join(layer_map), ", \n".join(parameter_map)))
    py_layers.write('import numpy as np\nimport _backend.nn as nn\nimport onnx.helper\nfrom onnx.backend.test.case.node import expect\nlayer_map = {}\ntensors = {}\n' + '\n\n'.join(py_layers_map))

    print(single_element)
    print(double_element)
    print(single_input)
    print(single_output)
    print(simple_element)
    print(complex_element)

    layer_map_file.close()


if (__name__ == "__main__"):
    onnx_proto()
  





'''
CategoryMap -> cat_strings-not Tensor is List()
Process TensorProto/Graph obj





'''