#include "../layer.h"
#ifndef IMPUTER_H
#define IMPUTER_H 
/*

    Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

input: Data to be processed.
output: Imputed output data
//*/
//Imputer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, float, int

//class stuff
namespace backend {   

    class Imputer : public Layer {
        typedef struct {
            Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
			Shape_t imputed_value_floats;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64; std::string imputed_value_floats;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Imputer(std::string n);
    
        void forward() { program->run(); }
        
        void init( Shape_t _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64); 
        void bind(std::string _imputed_value_floats, std::string _X_input, std::string _Y_output); 

        ~Imputer() {}

    };
    
}

#endif

