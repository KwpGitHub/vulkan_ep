#ifndef LEAKYRELU_H
#define LEAKYRELU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.

input: Input tensor
output: Output tensor
*/

//LeakyRelu
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float

//class stuff
namespace backend {   

    class LeakyRelu : public Layer {
        typedef struct {
            float alpha;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        float alpha;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LeakyRelu(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( float _alpha); 
        void bind(std::string _X_i, std::string _Y_o); 

        ~LeakyRelu() {}
    };

}

#endif

