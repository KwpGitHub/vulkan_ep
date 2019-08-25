#ifndef SUM_H
#define SUM_H 

#include "../layer.h"

/*

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for sum.
output: Output tensor.
*/

//Sum
//INPUTS:                   
//OPTIONAL_INPUTS:          x0_i, x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i, x9_i, x10_i, x11_i, x12_i, x13_i, x14_i, x15_i, x16_i, x17_i, x18_i, x19_i, x20_i, x21_i, x22_i, x23_i, x24_i, x25_i, x26_i, x27_i, x28_i, x29_i, x30_i, x31_i
//OUTPUS:                   sum_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Sum : public backend::Layer {
        typedef struct {          
            
            backend::Shape_t x0_i; backend::Shape_t x1_i; backend::Shape_t x2_i; backend::Shape_t x3_i; backend::Shape_t x4_i; backend::Shape_t x5_i; backend::Shape_t x6_i; backend::Shape_t x7_i; backend::Shape_t x8_i; backend::Shape_t x9_i; backend::Shape_t x10_i; backend::Shape_t x11_i; backend::Shape_t x12_i; backend::Shape_t x13_i; backend::Shape_t x14_i; backend::Shape_t x15_i; backend::Shape_t x16_i; backend::Shape_t x17_i; backend::Shape_t x18_i; backend::Shape_t x19_i; backend::Shape_t x20_i; backend::Shape_t x21_i; backend::Shape_t x22_i; backend::Shape_t x23_i; backend::Shape_t x24_i; backend::Shape_t x25_i; backend::Shape_t x26_i; backend::Shape_t x27_i; backend::Shape_t x28_i; backend::Shape_t x29_i; backend::Shape_t x30_i; backend::Shape_t x31_i;
            backend::Shape_t sum_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        
        std::string x0_i; std::string x1_i; std::string x2_i; std::string x3_i; std::string x4_i; std::string x5_i; std::string x6_i; std::string x7_i; std::string x8_i; std::string x9_i; std::string x10_i; std::string x11_i; std::string x12_i; std::string x13_i; std::string x14_i; std::string x15_i; std::string x16_i; std::string x17_i; std::string x18_i; std::string x19_i; std::string x20_i; std::string x21_i; std::string x22_i; std::string x23_i; std::string x24_i; std::string x25_i; std::string x26_i; std::string x27_i; std::string x28_i; std::string x29_i; std::string x30_i; std::string x31_i;
        std::string sum_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Sum(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _sum_o); 
        virtual void build();

        ~Sum() {}
    };
   
}
#endif

