#ifndef GEMM_H
#define GEMM_H 

#include "../layer.h"

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
namespace layers {   

    class Gemm : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_alpha; float m_beta; int m_transA; int m_transB;
        std::string m_A_i; std::string m_B_i; std::string m_C_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Gemm(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha,  float _beta,  int _transA,  int _transB); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _C_i, std::string _Y_o); 
        virtual void build();

        ~Gemm() {}
    };
   
}
#endif

