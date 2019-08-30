#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H 

#include "../layer.h"

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
namespace layers {   

    class HardSigmoid : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_alpha; float m_beta;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        HardSigmoid(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha,  float _beta); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~HardSigmoid() {}
    };
   
}
#endif

