#ifndef ACOSH_H
#define ACOSH_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Calculates the hyperbolic arccosine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic arccosine values of the input tensor computed element-wise
//*/
//Acosh
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

    class Acosh : public Layer {
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
        Acosh(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~Acosh() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Acosh::Acosh(std::string n) : Layer(n) { }
       
    vuh::Device* Acosh::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Acosh::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Acosh::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/acosh.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Acosh, Layer>(m, "Acosh")
            .def(py::init<std::string> ())
            .def("forward", &Acosh::forward)
            .def("init", &Acosh::init)
            .def("call", (void (Acosh::*) (std::string, std::string)) &Acosh::call);
    }
}

#endif

/* PYTHON STUFF

*/

