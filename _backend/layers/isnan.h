#ifndef ISNAN_H
#define ISNAN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
Returns which elements of the input are NaN.
input: input
output: output
*/

//IsNaN
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

    class IsNaN : public Layer {
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
        IsNaN();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~IsNaN() {}
    };

    
    void init_layer_IsNaN(py::module& m) {
        // py::class_(m, "IsNaN");
    }
    

}


#endif

