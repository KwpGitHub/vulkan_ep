#ifndef ZIPMAP_H
#define ZIPMAP_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

input: The input values
output: The output map
*/

//ZipMap
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*

//class stuff
namespace backend {   

    class ZipMap : public Layer {
        typedef struct {
            Shape_t classlabels_int64s;
			Shape_t classlabels_strings;
            Shape_t X_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_int64s; std::string classlabels_strings;
        std::string X_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ZipMap();
    
        void forward() { program->run(); }
        
        void init( Shape_t _classlabels_int64s); 
        void bind(std::string _classlabels_strings, std::string _X_input, std::string _Z_output); 

        ~ZipMap() {}
    };

    
    void init_layer_ZipMap(py::module& m) {
        // py::class_(m, "ZipMap");
    }
    

}


#endif

