#ifndef GEMM_H
#define GEMM_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
input: Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
input: Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
input: Input tensor C. The shape of C should be unidirectional broadcastable to (M, N).
output: Output tensor of shape (M, N).
*/

//Gemm
//INPUTS:                   A_i, B_i, C_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta, transA, transB
//OPTIONAL_PARAMETERS_TYPE: float, float, int, int


//class stuff
namespace backend {   

    class Gemm : public Layer {
        typedef struct {
            float alpha; float beta; int transA; int transB;
			
            Shape_t A_i; Shape_t B_i; Shape_t C_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        float alpha; float beta; int transA; int transB;
        std::string A_i; std::string B_i; std::string C_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Gemm(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _alpha,  float _beta,  int _transA,  int _transB); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _C_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gemm.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[C_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~Gemm() {}
    };
   
}
#endif

