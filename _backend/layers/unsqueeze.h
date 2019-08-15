#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

input: Original tensor
output: Reshaped tensor with same data as input.
//*/
//Unsqueeze
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Unsqueeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_input;
            
            Shape_t expanded_output;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_input;
        
        std::string expanded_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Unsqueeze(std::string n, Shape_t axes);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string expanded_output); 

        ~Unsqueeze() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Unsqueeze::Unsqueeze(std::string n, Shape_t axes) : Layer(n) { }
       
    vuh::Device* Unsqueeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Unsqueeze::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.expanded_output = tensor_dict[expanded_output]->shape();
 
		binding.axes = axes;
 
    }
    
    void Unsqueeze::call(std::string data_input, std::string expanded_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/unsqueeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[expanded_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Unsqueeze, Layer>(m, "Unsqueeze")
            .def(py::init<std::string, Shape_t> ())
            .def("forward", &Unsqueeze::forward)
            .def("init", &Unsqueeze::init)
            .def("call", (void (Unsqueeze::*) (std::string, std::string)) &Unsqueeze::call);
    }
}

#endif

/* PYTHON STUFF

*/

