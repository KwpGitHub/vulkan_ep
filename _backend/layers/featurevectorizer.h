#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.

input: An ordered collection of tensors, all with the same element type.
output: The output array, elements ordered as the inputs.
*/

//FeatureVectorizer
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      inputdimensions
//OPTIONAL_PARAMETERS_TYPE: Shape_t

//class stuff
namespace backend {   

    class FeatureVectorizer : public Layer {
        typedef struct {
            Shape_t inputdimensions;
			
            
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t inputdimensions;
        
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        FeatureVectorizer();
    
        void forward() { program->run(); }
        
        void init( Shape_t _inputdimensions); 
        void bind(std::string _Y_output); 

        ~FeatureVectorizer() {}
    };

    
    void init_layer_FeatureVectorizer(py::module& m) {
        // py::class_(m, "FeatureVectorizer");
    }
    

}


#endif

