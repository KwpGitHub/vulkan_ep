#ifndef CONSTANT_H
#define CONSTANT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
A constant tensor.

output: Output tensor containing the same value of the provided tensor.
//*/
//Constant
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Constant : public Layer {
        typedef struct {
            
			Shape_t value;
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        std::string value;
        
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Constant(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string value, std::string output_output); 

        ~Constant() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Constant::Constant(std::string n) : Layer(n) { }
       
    vuh::Device* Constant::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Constant::init() {      
    

		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.value = tensor_dict[value]->shape();
 
    }
    
    void Constant::call(std::string value, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Constant, Layer>(m, "Constant")
            .def(py::init<std::string> ())
            .def("forward", &Constant::forward)
            .def("init", &Constant::init)
            .def("call", (void (Constant::*) (std::string, std::string)) &Constant::call);
    }
}

#endif

/* PYTHON STUFF

*/

