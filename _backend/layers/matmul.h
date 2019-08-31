#ifndef MATMUL_H
#define MATMUL_H 

#include "../layer.h"

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
namespace layers {   

    class MatMul : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_A_i; std::string m_B_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        MatMul(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _Y_o); 
        virtual void build();

        ~MatMul() {}
    };
   
}
#endif

