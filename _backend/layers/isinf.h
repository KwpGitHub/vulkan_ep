#ifndef ISINF_H
#define ISINF_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
Map infinity to true and other values to false.
input: input
output: output
//*/
//IsInf
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      detect_negative, detect_positive
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class IsInf : public Layer {
        typedef struct {
            int detect_negative; int detect_positive;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int detect_negative; int detect_positive;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        IsInf(std::string n, int detect_negative, int detect_positive);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~IsInf() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    IsInf::IsInf(std::string n, int detect_negative, int detect_positive) : Layer(n) { }
       
    vuh::Device* IsInf::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void IsInf::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.detect_negative = detect_negative;
  		binding.detect_positive = detect_positive;
 
    }
    
    void IsInf::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/isinf.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<IsInf, Layer>(m, "IsInf")
            .def(py::init<std::string, int, int> ())
            .def("forward", &IsInf::forward)
            .def("init", &IsInf::init)
            .def("call", (void (IsInf::*) (std::string, std::string)) &IsInf::call);
    }
}

#endif

/* PYTHON STUFF

*/

