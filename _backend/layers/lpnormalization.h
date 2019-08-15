#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Given a matrix, apply Lp-normalization along the provided axis.

input: Input matrix
output: Matrix after normalization
//*/
//LpNormalization
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class LpNormalization : public Layer {
        typedef struct {
            int axis; int p;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis; int p;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpNormalization(std::string n, int axis, int p);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~LpNormalization() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LpNormalization::LpNormalization(std::string n, int axis, int p) : Layer(n) { }
       
    vuh::Device* LpNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LpNormalization::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
  		binding.p = p;
 
    }
    
    void LpNormalization::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lpnormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LpNormalization, Layer>(m, "LpNormalization")
            .def(py::init<std::string, int, int> ())
            .def("forward", &LpNormalization::forward)
            .def("init", &LpNormalization::init)
            .def("call", (void (LpNormalization::*) (std::string, std::string)) &LpNormalization::call);
    }
}

#endif

/* PYTHON STUFF

*/

