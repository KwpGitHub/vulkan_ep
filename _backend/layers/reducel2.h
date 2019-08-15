#ifndef REDUCEL2_H
#define REDUCEL2_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes the L2 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.

*/
//ReduceL2
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

    class ReduceL2 : public Layer {
        typedef struct {    
            Shape_t axes; int keepdims;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* reduced_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t axes; int keepdims;
		
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
        ReduceL2(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~ReduceL2() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ReduceL2::ReduceL2(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducel2.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* ReduceL2::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ReduceL2::init() {
		binding.data_input = input.data_input->shape();
 
		binding.reduced_output = output.reduced_output->shape();
 
		binding.axes = parameters.axes;
  		binding.keepdims = parameters.keepdims;
 
        program->bind(binding, *input.data_input->data(), *output.reduced_output->data());
    }
    
    void ReduceL2::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceL2, Layer>(m, "ReduceL2")
            .def("forward", &ReduceL2::forward);    
    }
}*/

#endif
