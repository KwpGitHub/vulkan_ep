#ifndef RESHAPE_H
#define RESHAPE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).
input: An input tensor.
input: Specified shape for output.
output: Reshaped data.
//*/
//Reshape
//INPUTS:                   data_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Reshape : public Layer {
        typedef struct {
            
			
            Shape_t data_input; Shape_t shape_input;
            
            Shape_t reshaped_output;
            
        } binding_descriptor;

        
        std::string data_input; std::string shape_input;
        
        std::string reshaped_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Reshape(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string shape_input, std::string reshaped_output); 

        ~Reshape() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Reshape::Reshape(std::string n) : Layer(n) { }
       
    vuh::Device* Reshape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Reshape::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.shape_input = tensor_dict[shape_input]->shape();
 
		binding.reshaped_output = tensor_dict[reshaped_output]->shape();
 

    }
    
    void Reshape::call(std::string data_input, std::string shape_input, std::string reshaped_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reshape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[shape_input]->data(), *tensor_dict[reshaped_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Reshape, Layer>(m, "Reshape")
            .def(py::init<std::string> ())
            .def("forward", &Reshape::forward)
            .def("init", &Reshape::init)
            .def("call", (void (Reshape::*) (std::string, std::string, std::string)) &Reshape::call);
    }
}

#endif

/* PYTHON STUFF

*/

