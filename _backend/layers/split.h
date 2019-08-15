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
//*/
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
			
            Shape_t input_input;
            
            
            
        } binding_descriptor;

        int axis; Shape_t split;
        std::string input_input;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Split(std::string n, int axis, Shape_t split);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input); 

        ~Split() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Split::Split(std::string n, int axis, Shape_t split) : Layer(n) { }
       
    vuh::Device* Split::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Split::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 

		binding.axis = axis;
  		binding.split = split;
 
    }
    
    void Split::call(std::string input_input){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Split, Layer>(m, "Split")
            .def(py::init<std::string, int, Shape_t> ())
            .def("forward", &Split::forward)
            .def("init", &Split::init)
            .def("call", (void (Split::*) (std::string)) &Split::call);
    }
}

#endif

/* PYTHON STUFF

*/

