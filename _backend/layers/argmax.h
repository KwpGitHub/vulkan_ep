#ifndef ARGMAX_H
#define ARGMAX_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes the indices of the max elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.

*/
//ArgMax
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

    class ArgMax : public Layer {
        typedef struct {    
            int axis; int keepdims;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* reduced_output;
            
        } output_descriptor;

        typedef struct {
            int axis; int keepdims;
		
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArgMax(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~ArgMax() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ArgMax::ArgMax(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmax.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* ArgMax::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArgMax::init() {
		binding.data_input = input.data_input->shape();
 
		binding.reduced_output = output.reduced_output->shape();
 
		binding.axis = parameters.axis;
  		binding.keepdims = parameters.keepdims;
 
        program->bind(binding, *input.data_input->data(), *output.reduced_output->data());
    }
    
    void ArgMax::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ArgMax, Layer>(m, "ArgMax")
            .def("forward", &ArgMax::forward);    
    }
}*/

#endif
