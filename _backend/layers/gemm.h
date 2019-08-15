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
//*/
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
			
            Shape_t A_input; Shape_t B_input; Shape_t C_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha; float beta; int transA; int transB;
        std::string A_input; std::string B_input; std::string C_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Gemm(std::string n, float alpha, float beta, int transA, int transB);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string A_input, std::string B_input, std::string C_input, std::string Y_output); 

        ~Gemm() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Gemm::Gemm(std::string n, float alpha, float beta, int transA, int transB) : Layer(n) { }
       
    vuh::Device* Gemm::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Gemm::init() {      
    
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.C_input = tensor_dict[C_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.transA = transA;
  		binding.transB = transB;
 
    }
    
    void Gemm::call(std::string A_input, std::string B_input, std::string C_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gemm.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[C_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Gemm, Layer>(m, "Gemm")
            .def(py::init<std::string, float, float, int, int> ())
            .def("forward", &Gemm::forward)
            .def("init", &Gemm::init)
            .def("call", (void (Gemm::*) (std::string, std::string, std::string, std::string)) &Gemm::call);
    }
}

#endif

/* PYTHON STUFF

*/

