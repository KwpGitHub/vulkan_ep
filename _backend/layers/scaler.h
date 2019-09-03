#pragma once
#ifndef SCALER_H
#define SCALER_H 

#include "../layer.h"

/*

    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

input: Data to be scaled.
output: Scaled output data.
*/

//Scaler
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>


//class stuff
namespace layers {   

    class Scaler : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_offset; std::vector<float> m_scale;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Scaler(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _offset,  std::vector<float> _scale); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Scaler() {}
    };
   
}
#endif

