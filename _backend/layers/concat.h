#ifndef CONCAT_H
#define CONCAT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
Concatenate a list of tensors into a single tensor
input: List of tensors for concatenation
output: Concatenated tensor
*/

//Concat
//INPUTS:                   
//OPTIONAL_INPUTS:          x0_i, x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i, x9_i, x10_i, x11_i, x12_i, x13_i, x14_i, x15_i, x16_i, x17_i, x18_i, x19_i, x20_i, x21_i, x22_i, x23_i, x24_i, x25_i, x26_i, x27_i, x28_i, x29_i, x30_i, x31_i
//OUTPUS:                   concat_result_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axis
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace backend {   

    class Concat : public Layer {
        typedef struct {
            int axis;
			
            
            Shape_t x0_i; Shape_t x1_i; Shape_t x2_i; Shape_t x3_i; Shape_t x4_i; Shape_t x5_i; Shape_t x6_i; Shape_t x7_i; Shape_t x8_i; Shape_t x9_i; Shape_t x10_i; Shape_t x11_i; Shape_t x12_i; Shape_t x13_i; Shape_t x14_i; Shape_t x15_i; Shape_t x16_i; Shape_t x17_i; Shape_t x18_i; Shape_t x19_i; Shape_t x20_i; Shape_t x21_i; Shape_t x22_i; Shape_t x23_i; Shape_t x24_i; Shape_t x25_i; Shape_t x26_i; Shape_t x27_i; Shape_t x28_i; Shape_t x29_i; Shape_t x30_i; Shape_t x31_i;
            Shape_t concat_result_o;
            
        } binding_descriptor;

        int axis;
        
        std::string x0_i; std::string x1_i; std::string x2_i; std::string x3_i; std::string x4_i; std::string x5_i; std::string x6_i; std::string x7_i; std::string x8_i; std::string x9_i; std::string x10_i; std::string x11_i; std::string x12_i; std::string x13_i; std::string x14_i; std::string x15_i; std::string x16_i; std::string x17_i; std::string x18_i; std::string x19_i; std::string x20_i; std::string x21_i; std::string x22_i; std::string x23_i; std::string x24_i; std::string x25_i; std::string x26_i; std::string x27_i; std::string x28_i; std::string x29_i; std::string x30_i; std::string x31_i;
        std::string concat_result_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Concat(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _axis); 
        virtual void bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _concat_result_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/concat.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[x0_i]->data(), *tensor_dict[x1_i]->data(), *tensor_dict[x2_i]->data(), *tensor_dict[x3_i]->data(), *tensor_dict[x4_i]->data(), *tensor_dict[x5_i]->data(), *tensor_dict[x6_i]->data(), *tensor_dict[x7_i]->data(), *tensor_dict[x8_i]->data(), *tensor_dict[x9_i]->data(), *tensor_dict[x10_i]->data(), *tensor_dict[x11_i]->data(), *tensor_dict[x12_i]->data(), *tensor_dict[x13_i]->data(), *tensor_dict[x14_i]->data(), *tensor_dict[x15_i]->data(), *tensor_dict[x16_i]->data(), *tensor_dict[x17_i]->data(), *tensor_dict[x18_i]->data(), *tensor_dict[x19_i]->data(), *tensor_dict[x20_i]->data(), *tensor_dict[x21_i]->data(), *tensor_dict[x22_i]->data(), *tensor_dict[x23_i]->data(), *tensor_dict[x24_i]->data(), *tensor_dict[x25_i]->data(), *tensor_dict[x26_i]->data(), *tensor_dict[x27_i]->data(), *tensor_dict[x28_i]->data(), *tensor_dict[x29_i]->data(), *tensor_dict[x30_i]->data(), *tensor_dict[x31_i]->data(), *tensor_dict[concat_result_o]->data());
        }

        ~Concat() {}
    };
   
}
#endif

