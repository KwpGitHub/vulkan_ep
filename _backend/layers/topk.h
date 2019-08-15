#ifndef TOPK_H
#define TOPK_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
   
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.

input: Tensor of shape [a_1, a_2, ..., a_n, r]
input: A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
output: Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing top K values from the input tensor
output: Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing the corresponding input tensor indices for the top K values.
//*/
//TopK
//INPUTS:                   X_input, K_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Values_output, Indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class TopK : public Layer {
        typedef struct {
            int axis;
			
            Shape_t X_input; Shape_t K_input;
            
            Shape_t Values_output; Shape_t Indices_output;
            
        } binding_descriptor;

        int axis;
        std::string X_input; std::string K_input;
        
        std::string Values_output; std::string Indices_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TopK(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string K_input, std::string Values_output, std::string Indices_output); 

        ~TopK() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    TopK::TopK(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* TopK::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TopK::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.K_input = tensor_dict[K_input]->shape();
 
		binding.Values_output = tensor_dict[Values_output]->shape();
  		binding.Indices_output = tensor_dict[Indices_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void TopK::call(std::string X_input, std::string K_input, std::string Values_output, std::string Indices_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/topk.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[K_input]->data(), *tensor_dict[Values_output]->data(), *tensor_dict[Indices_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<TopK, Layer>(m, "TopK")
            .def(py::init<std::string, int> ())
            .def("forward", &TopK::forward)
            .def("init", &TopK::init)
            .def("call", (void (TopK::*) (std::string, std::string, std::string, std::string)) &TopK::call);
    }
}

#endif

/* PYTHON STUFF

*/

