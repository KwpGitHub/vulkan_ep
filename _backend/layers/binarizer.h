#ifndef BINARIZER_H
#define BINARIZER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

input: Data to be binarized
output: Binarized output data
*/

//Binarizer
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      threshold
//OPTIONAL_PARAMETERS_TYPE: float

//class stuff
namespace backend {   

    class Binarizer : public Layer {
        typedef struct {
            float threshold;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        float threshold;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Binarizer(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( float _threshold); 
        void bind(std::string _X_i, std::string _Y_o); 

        ~Binarizer() {}
    };

}

#endif

