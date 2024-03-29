#pragma once
#ifndef IMPUTER_H
#define IMPUTER_H 

#include "../layer.h"

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
*/

//Imputer
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<int>, float, int


//class stuff
namespace layers {   

    class Imputer : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_imputed_value_floats; std::vector<int> m_imputed_value_int64s; float m_replaced_value_float; int m_replaced_value_int64;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Imputer(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _imputed_value_floats,  std::vector<int> _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Imputer() {}
    };
   
}
#endif

