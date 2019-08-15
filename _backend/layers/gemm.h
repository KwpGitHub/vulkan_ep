#ifndef GEMM_H
#define GEMM_H 
#include <pybind11/pybind11.h>
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
//INPUTS:                   A_input, B_input, C_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta, transA, transB
//OPTIONAL_PARAMETERS_TYPE: float, float, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Gemm : public Layer {
        typedef struct {    
            float alpha; float beta; int transA; int transB;
        } parameter_descriptor;  

        typedef struct {
            Tensor* A_input; Tensor* B_input; Tensor* C_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            float alpha; float beta; int transA; int transB;
		
            Shape_t A_input; Shape_t B_input; Shape_t C_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Gemm(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Gemm() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Gemm::Gemm(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gemm.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Gemm::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Gemm::init() {
		binding.A_input = input.A_input->shape();
  		binding.B_input = input.B_input->shape();
  		binding.C_input = input.C_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.alpha = parameters.alpha;
  		binding.beta = parameters.beta;
  		binding.transA = parameters.transA;
  		binding.transB = parameters.transB;
 
        program->bind(binding, *input.A_input->data(), *input.B_input->data(), *input.C_input->data(), *output.Y_output->data());
    }
    
    void Gemm::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Gemm, Layer>(m, "Gemm")
            .def("forward", &Gemm::forward);    
    }
}*/

#endif
