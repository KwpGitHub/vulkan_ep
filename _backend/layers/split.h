#ifndef SPLIT_H
#define SPLIT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.

input: The tensor to split
output: One or more outputs forming list of tensors after splitting

*/
//Split
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class Split : public Layer {
        typedef struct {    
            int axis; Shape_t split;
        } parameter_descriptor;  

        typedef struct {
            Tensor* input_input;
            
        } input_desriptor;

        typedef struct {
            
            
        } output_descriptor;

        typedef struct {
            int axis; Shape_t split;
		
            Shape_t input_input;
            
            
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Split(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Split() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Split::Split(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Split::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Split::init() {
		binding.input_input = input.input_input->shape();
 

		binding.axis = parameters.axis;
  		binding.split = parameters.split;
 
        program->bind(binding, *input.input_input->data());
    }
    
    void Split::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Split, Layer>(m, "Split")
            .def("forward", &Split::forward);    
    }
}*/

#endif
