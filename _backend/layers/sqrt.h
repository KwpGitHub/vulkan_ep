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
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sqrt : public Layer {
        typedef struct {
            
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sqrt();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~Sqrt() {}
    };

    
    void init_layer_Sqrt(py::module& m) {
        // py::class_(m, "Sqrt");
    }
    

}


#endif

