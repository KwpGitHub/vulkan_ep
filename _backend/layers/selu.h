#ifndef SELU_H
#define SELU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//Selu
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, gamma
//OPTIONAL_PARAMETERS_TYPE: float, float

//class stuff
namespace backend {   

    class Selu : public Layer {
        typedef struct {
            float alpha; float gamma;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha; float gamma;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Selu();
    
        void forward() { program->run(); }
        
        void init( float _alpha,  float _gamma); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~Selu() {}
    };

    
    void init_layer_Selu(py::module& m) {
        // py::class_(m, "Selu");
    }
    

}


#endif

