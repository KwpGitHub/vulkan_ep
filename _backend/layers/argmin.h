#ifndef ARGMIN_H
#define ARGMIN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes the indices of the min elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.
//*/
//ArgMin
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class ArgMin : public Layer {
        typedef struct {
            int axis; int keepdims;
			
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        int axis; int keepdims;
        std::string data_input;
        
        std::string reduced_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArgMin(std::string n, int axis, int keepdims);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string reduced_output); 

        ~ArgMin() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ArgMin::ArgMin(std::string n, int axis, int keepdims) : Layer(n) { }
       
    vuh::Device* ArgMin::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArgMin::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.reduced_output = tensor_dict[reduced_output]->shape();
 
		binding.axis = axis;
  		binding.keepdims = keepdims;
 
    }
    
    void ArgMin::call(std::string data_input, std::string reduced_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmin.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[reduced_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ArgMin, Layer>(m, "ArgMin")
            .def(py::init<std::string, int, int> ())
            .def("forward", &ArgMin::forward)
            .def("init", &ArgMin::init)
            .def("call", (void (ArgMin::*) (std::string, std::string)) &ArgMin::call);
    }
}

#endif

/* PYTHON STUFF

*/

