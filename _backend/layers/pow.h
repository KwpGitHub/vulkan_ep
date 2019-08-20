#ifndef POW_H
#define POW_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
input: First operand, base of the exponent.
input: Second operand, power of the exponent.
output: Output tensor (same size as X)
*/

//Pow
//INPUTS:                   X_i, Y_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Pow : public Layer {
        typedef struct {
            
			
            Shape_t X_i; Shape_t Y_i;
            
            Shape_t Z_o;
            
        } binding_descriptor;

        
        std::string X_i; std::string Y_i;
        
        std::string Z_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pow(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_i, std::string _Y_i, std::string _Z_o); 

        ~Pow() {}
    };

}

#endif

