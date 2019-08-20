#ifndef MATMUL_H
#define MATMUL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

input: N-dimensional matrix A
input: N-dimensional matrix B
output: Matrix multiply results from A * B
*/

//MatMul
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class MatMul : public Layer {
        typedef struct {
            
			
            Shape_t A_i; Shape_t B_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        
        std::string A_i; std::string B_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMul(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _A_i, std::string _B_i, std::string _Y_o); 

        ~MatMul() {}
    };

}

#endif

