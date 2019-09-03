#pragma once
#ifndef MEAN_H
#define MEAN_H 

#include "../layer.h"

/*

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for mean.
output: Output tensor.
*/

//Mean
//INPUTS:                   
//OPTIONAL_INPUTS:          x0_i, x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i, x9_i, x10_i, x11_i, x12_i, x13_i, x14_i, x15_i, x16_i, x17_i, x18_i, x19_i, x20_i, x21_i, x22_i, x23_i, x24_i, x25_i, x26_i, x27_i, x28_i, x29_i, x30_i, x31_i
//OUTPUS:                   mean_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Mean : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        
        std::string m_x0_i; std::string m_x1_i; std::string m_x2_i; std::string m_x3_i; std::string m_x4_i; std::string m_x5_i; std::string m_x6_i; std::string m_x7_i; std::string m_x8_i; std::string m_x9_i; std::string m_x10_i; std::string m_x11_i; std::string m_x12_i; std::string m_x13_i; std::string m_x14_i; std::string m_x15_i; std::string m_x16_i; std::string m_x17_i; std::string m_x18_i; std::string m_x19_i; std::string m_x20_i; std::string m_x21_i; std::string m_x22_i; std::string m_x23_i; std::string m_x24_i; std::string m_x25_i; std::string m_x26_i; std::string m_x27_i; std::string m_x28_i; std::string m_x29_i; std::string m_x30_i; std::string m_x31_i;
        std::string m_mean_o;
        

        binding_descriptor   binding;
       

    public:
        Mean(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _mean_o); 
        virtual void build();

        ~Mean() {}
    };
   
}
#endif

