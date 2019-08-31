#ifndef NORMALIZER_H
#define NORMALIZER_H 

#include "../layer.h"

/*

    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.

input: Data to be encoded, a tensor of shape [N,C] or [C]
output: Encoded output data
*/

//Normalizer
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      norm
//OPTIONAL_PARAMETERS_TYPE: std::string


//class stuff
namespace layers {   

    class Normalizer : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string m_norm;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Normalizer(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _norm); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Normalizer() {}
    };
   
}
#endif

