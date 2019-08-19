#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//HardSigmoid
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float

//class stuff
namespace backend {   

    class HardSigmoid : public Layer {
        typedef struct {
            float alpha; float beta;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha; float beta;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        HardSigmoid();
    
        void forward() { program->run(); }
        
        void init( float _alpha,  float _beta); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~HardSigmoid() {}
    };

    
    void init_layer_HardSigmoid(py::module& m) {
        // py::class_(m, "HardSigmoid");
    }
    

}


#endif

