#ifndef REDUCEMIN_H
#define REDUCEMIN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes the min of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
//*/
//ReduceMin
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class ReduceMin : public Layer {
        typedef struct {
            Shape_t axes; int keepdims;
			
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        Shape_t axes; int keepdims;
        std::string data_input;
        
        std::string reduced_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReduceMin(std::string n, Shape_t axes, int keepdims);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string reduced_output); 

        ~ReduceMin() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ReduceMin::ReduceMin(std::string n, Shape_t axes, int keepdims) : Layer(n) { }
       
    vuh::Device* ReduceMin::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ReduceMin::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.reduced_output = tensor_dict[reduced_output]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 
    }
    
    void ReduceMin::call(std::string data_input, std::string reduced_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducemin.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[reduced_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceMin, Layer>(m, "ReduceMin")
            .def(py::init<std::string, Shape_t, int> ())
            .def("forward", &ReduceMin::forward)
            .def("init", &ReduceMin::init)
            .def("call", (void (ReduceMin::*) (std::string, std::string)) &ReduceMin::call);
    }
}

#endif

/* PYTHON STUFF

*/

