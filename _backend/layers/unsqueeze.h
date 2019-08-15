#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

input: Original tensor
output: Reshaped tensor with same data as input.

*/
//Unsqueeze
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Unsqueeze : public Layer {
        typedef struct {    
            Shape_t axes;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* expanded_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t axes;
		
            Shape_t data_input;
            
            Shape_t expanded_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Unsqueeze(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Unsqueeze() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Unsqueeze::Unsqueeze(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/unsqueeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Unsqueeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Unsqueeze::init() {
		binding.data_input = input.data_input->shape();
 
		binding.expanded_output = output.expanded_output->shape();
 
		binding.axes = parameters.axes;
 
        program->bind(binding, *input.data_input->data(), *output.expanded_output->data());
    }
    
    void Unsqueeze::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Unsqueeze, Layer>(m, "Unsqueeze")
            .def("forward", &Unsqueeze::forward);    
    }
}*/

#endif
