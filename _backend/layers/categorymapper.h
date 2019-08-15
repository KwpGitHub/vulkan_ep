#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending 
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.

input: Input data
output: Output data. If strings are input, the output values are integers, and vice versa.

*/
//CategoryMapper
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, default_int64, default_string
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class CategoryMapper : public Layer {
        typedef struct {    
            Shape_t cats_int64s; Tensor* cats_strings; int default_int64; int default_string;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t cats_int64s; int default_int64; int default_string;
		Shape_t cats_strings;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        CategoryMapper(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~CategoryMapper() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    CategoryMapper::CategoryMapper(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/categorymapper.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* CategoryMapper::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void CategoryMapper::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.cats_int64s = parameters.cats_int64s;
  		binding.default_int64 = parameters.default_int64;
  		binding.default_string = parameters.default_string;
  		binding.cats_strings = parameters.cats_strings->shape();
 
        program->bind(binding, *parameters.cats_strings->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void CategoryMapper::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<CategoryMapper, Layer>(m, "CategoryMapper")
            .def("forward", &CategoryMapper::forward);    
    }
}*/

#endif
