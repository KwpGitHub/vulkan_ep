#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class BatchNormalization : public Layer {
        typedef struct {
            float epsilon; float momentum;
			
            Shape_t X_i; Shape_t scale_i; Shape_t B_i; Shape_t mean_i; Shape_t var_i;
            
            Shape_t Y_o;
            Shape_t mean_o; Shape_t var_o; Shape_t saved_mean_o; Shape_t saved_var_o;
        } binding_descriptor;

        float epsilon; float momentum;
        std::string X_i; std::string scale_i; std::string B_i; std::string mean_i; std::string var_i;
        
        std::string Y_o;
        std::string mean_o; std::string var_o; std::string saved_mean_o; std::string saved_var_o;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        BatchNormalization(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _epsilon,  float _momentum); 
        virtual void bind(std::string _X_i, std::string _scale_i, std::string _B_i, std::string _mean_i, std::string _var_i, std::string _Y_o, std::string _mean_o, std::string _var_o, std::string _saved_mean_o, std::string _saved_var_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/batchnormalization.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[scale_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[mean_i]->data(), *tensor_dict[var_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[mean_o]->data(), *tensor_dict[var_o]->data(), *tensor_dict[saved_mean_o]->data(), *tensor_dict[saved_var_o]->data());
        }

        ~BatchNormalization() {}
    };
   
}
#endif

