#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//HardSigmoid
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace backend {   

    class HardSigmoid : public Layer {
        typedef struct {
            float alpha; float beta;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        float alpha; float beta;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        HardSigmoid(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _alpha,  float _beta); 
        virtual void bind(std::string _X_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/hardsigmoid.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~HardSigmoid() {}
    };
   
}
#endif

