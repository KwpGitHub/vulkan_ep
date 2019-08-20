#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count 
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8, 
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.

input: Data to be encoded.
output: Encoded output data, having one more dimension than X.
*/

//OneHotEncoder
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, zeros
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int

//class stuff
namespace backend {   

    class OneHotEncoder : public Layer {
        typedef struct {
            Shape_t cats_int64s; int zeros;
			Shape_t cats_strings;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t cats_int64s; int zeros; std::string cats_strings;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        OneHotEncoder(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _cats_int64s,  int _zeros); 
        void bind(std::string _cats_strings, std::string _X_input, std::string _Y_output); 

        ~OneHotEncoder() {}
    };

}

#endif

