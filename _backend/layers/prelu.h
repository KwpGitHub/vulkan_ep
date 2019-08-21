#ifndef PRELU_H
#define PRELU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
input: Input tensor
input: Slope tensor. The shape of slope can be smaller then first input X; if so, its shape must be unidirectional broadcastable to X
output: Output tensor (same size as X)
*/

//PRelu
//INPUTS:                   X_i, slope_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace backend {   

    class PRelu : public Layer {
        typedef struct {
            
			
            Shape_t X_i; Shape_t slope_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        
        std::string X_i; std::string slope_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        PRelu(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _slope_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/prelu.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[slope_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~PRelu() {}
    };
   
}
#endif

