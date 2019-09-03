#pragma once
#ifndef ZIPMAP_H
#define ZIPMAP_H 

#include "../layer.h"

/*

    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

input: The input values
output: The output map
*/

//ZipMap
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>


//class stuff
namespace layers {   

    class ZipMap : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> m_classlabels_int64s; std::vector<std::string> m_classlabels_strings;
        std::string m_X_i;
        
        std::string m_Z_o;
        

        binding_descriptor   binding;
       

    public:
        ZipMap(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings); 
        virtual void bind(std::string _X_i, std::string _Z_o); 
        virtual void build();

        ~ZipMap() {}
    };
   
}
#endif

