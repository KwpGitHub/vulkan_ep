#ifndef SQRT_H
#define SQRT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.

input: Input tensor
output: Output tensor
*/

//Sqrt
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sqrt : public Layer {
        typedef struct {
            
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sqrt(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_i, std::string _Y_o); 

        ~Sqrt() {}
    };

}

#endif

