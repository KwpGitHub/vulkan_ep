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
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class MatMul : public Layer {
        typedef struct {
            
			
            Shape_t A_input; Shape_t B_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string A_input; std::string B_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMul();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _A_input, std::string _B_input, std::string _Y_output); 

        ~MatMul() {}
    };

    
    void init_layer_MatMul(py::module& m) {
        // py::class_(m, "MatMul");
    }
    

}


#endif

