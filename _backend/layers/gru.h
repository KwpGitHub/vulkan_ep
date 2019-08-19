#ifndef GRU_H
#define GRU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: The input sequences packed (and potentially padded) into one 3-D tensor with the shape of `[seq_length, batch_size, input_size]`.
input: The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 3*hidden_size, input_size]`.
input: The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 3*hidden_size, hidden_size]`.
input: The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and `[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed to be 0
input: Optional tensor specifying lengths of the sequences in a batch. If not specified - assumed all sequences in the batch to have length `seq_length`. It has shape `[batch_size]`.
input: Optional initial value of the hidden. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`.
output: A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. 
output: The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`.
*/

//GRU
//INPUTS:                   X_input, W_input, R_input
//OPTIONAL_INPUTS:          B_input_opt, sequence_lens_input_opt, initial_h_input_opt
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_output_opt, Y_h_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int, int

//class stuff
namespace backend {   

    class GRU : public Layer {
        typedef struct {
            float clip; int direction; int hidden_size; int linear_before_reset;
			Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
            Shape_t X_input; Shape_t W_input; Shape_t R_input;
            Shape_t B_input_opt; Shape_t sequence_lens_input_opt; Shape_t initial_h_input_opt;
            
            Shape_t Y_output_opt; Shape_t Y_h_output_opt;
        } binding_descriptor;

        float clip; int direction; int hidden_size; int linear_before_reset; std::string activation_alpha; std::string activation_beta; std::string activations;
        std::string X_input; std::string W_input; std::string R_input;
        std::string B_input_opt; std::string sequence_lens_input_opt; std::string initial_h_input_opt;
        
        std::string Y_output_opt; std::string Y_h_output_opt;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        GRU();
    
        void forward() { program->run(); }
        
        void init( float _clip,  int _direction,  int _hidden_size,  int _linear_before_reset); 
        void bind(std::string _activation_alpha, std::string _activation_beta, std::string _activations, std::string _X_input, std::string _W_input, std::string _R_input, std::string _B_input_opt, std::string _sequence_lens_input_opt, std::string _initial_h_input_opt, std::string _Y_output_opt, std::string _Y_h_output_opt); 

        ~GRU() {}
    };

    
    void init_layer_GRU(py::module& m) {
        // py::class_(m, "GRU");
    }
    

}


#endif

