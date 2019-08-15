#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes the mean of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.

*/
//ReduceMean
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

    class ReduceMean : public Layer {
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
        ReduceMean(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~ReduceMean() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ReduceMean::ReduceMean(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducemean.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* ReduceMean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ReduceMean::init() {
		binding.data_input = input.data_input->shape();
 
		binding.reduced_output = output.reduced_output->shape();
 
		binding.axes = parameters.axes;
  		binding.keepdims = parameters.keepdims;
 
        program->bind(binding, *input.data_input->data(), *output.reduced_output->data());
    }
    
    void ReduceMean::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceMean, Layer>(m, "ReduceMean")
            .def("forward", &ReduceMean::forward);    
    }
}*/

#endif
