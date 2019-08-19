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
//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Pow : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t Y_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string Y_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pow();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _Y_input, std::string _Z_output); 

        ~Pow() {}
    };

    
    void init_layer_Pow(py::module& m) {
        // py::class_(m, "Pow");
    }
    

}


#endif

