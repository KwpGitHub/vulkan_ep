#ifndef LEAKYRELU_H
#define LEAKYRELU_H 

#include "../layer.h"

/*

LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.

input: Input tensor
output: Output tensor
*/

//LeakyRelu
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class LeakyRelu : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_alpha;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        LeakyRelu(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~LeakyRelu() {}
    };
   
}
#endif

