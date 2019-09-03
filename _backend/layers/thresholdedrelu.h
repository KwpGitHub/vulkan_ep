#pragma once
#ifndef THRESHOLDEDRELU_H
#define THRESHOLDEDRELU_H 

#include "../layer.h"

/*

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//ThresholdedRelu
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class ThresholdedRelu : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_alpha;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        ThresholdedRelu(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~ThresholdedRelu() {}
    };
   
}
#endif

