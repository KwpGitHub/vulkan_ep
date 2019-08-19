#ifndef SCALER_H
#define SCALER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

input: Data to be scaled.
output: Scaled output data.
*/

//Scaler
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*

//class stuff
namespace backend {   

    class Scaler : public Layer {
        typedef struct {
            
			Shape_t offset; Shape_t scale;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        std::string offset; std::string scale;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Scaler();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _offset, std::string _scale, std::string _X_input, std::string _Y_output); 

        ~Scaler() {}
    };

    
    void init_layer_Scaler(py::module& m) {
        // py::class_(m, "Scaler");
    }
    

}


#endif

