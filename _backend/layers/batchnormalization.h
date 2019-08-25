#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H 

#include "../layer.h"

/*

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
input: Scale tensor of shape (C).
input: Bias tensor of shape (C).
input: running (training) or estimated (testing) mean tensor of shape (C).
input: running (training) or estimated (testing) variance tensor of shape (C).
output: The output tensor of the same shape as X
output: The running mean after the BatchNormalization operator.
output: The running variance after the BatchNormalization operator.
output: Saved mean used during training to speed up gradient computation.
output: Saved variance used during training to speed up gradient computation.
*/

//BatchNormalization
//INPUTS:                   X_i, scale_i, B_i, mean_i, var_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         mean_o, var_o, saved_mean_o, saved_var_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon, momentum
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace layers {   

    class BatchNormalization : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i; backend::Shape_t scale_i; backend::Shape_t B_i; backend::Shape_t mean_i; backend::Shape_t var_i;
            
            backend::Shape_t Y_o;
            backend::Shape_t mean_o; backend::Shape_t var_o; backend::Shape_t saved_mean_o; backend::Shape_t saved_var_o;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float epsilon; float momentum;
        std::string X_i; std::string scale_i; std::string B_i; std::string mean_i; std::string var_i;
        
        std::string Y_o;
        std::string mean_o; std::string var_o; std::string saved_mean_o; std::string saved_var_o;

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        BatchNormalization(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init( float _epsilon,  float _momentum); 
        virtual void bind(std::string _X_i, std::string _scale_i, std::string _B_i, std::string _mean_i, std::string _var_i, std::string _Y_o, std::string _mean_o, std::string _var_o, std::string _saved_mean_o, std::string _saved_var_o); 
        virtual void build();

        ~BatchNormalization() {}
    };
   
}
#endif

