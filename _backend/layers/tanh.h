#ifndef TANH_H
#define TANH_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Calculates the hyperbolic tangent of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic tangent values of the input tensor computed element-wise
//*/
//Tanh
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Tanh : public Layer {
        typedef struct {
            
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Tanh(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~Tanh() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Tanh::Tanh(std::string n) : Layer(n) { }
       
    vuh::Device* Tanh::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Tanh::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Tanh::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tanh.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Tanh, Layer>(m, "Tanh")
            .def(py::init<std::string> ())
            .def("forward", &Tanh::forward)
            .def("init", &Tanh::init)
            .def("call", (void (Tanh::*) (std::string, std::string)) &Tanh::call);
    }
}

#endif

/* PYTHON STUFF

*/

