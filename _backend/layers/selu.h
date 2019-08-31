#ifndef SELU_H
#define SELU_H 

#include "../layer.h"

/*

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//Selu
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, gamma
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace layers {   

    class Selu : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_alpha; float m_gamma;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Selu(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha,  float _gamma); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Selu() {}
    };
   
}
#endif

