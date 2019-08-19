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
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
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
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float threshold;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Binarizer();
    
        void forward() { program->run(); }
        
        void init( float _threshold); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~Binarizer() {}
    };

    
    void init_layer_Binarizer(py::module& m) {
        // py::class_(m, "Binarizer");
    }
    

}


#endif

