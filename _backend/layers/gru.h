#ifndef GRU_H
#define GRU_H 

#include "../layer.h"

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
//INPUTS:                   X_i, W_i, R_i
//OPTIONAL_INPUTS:          B_i, sequence_lens_i, initial_h_i
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_o, Y_h_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>, std::vector<std::string>, float, std::string, int, int


//class stuff
namespace layers {   

    class GRU : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_activation_alpha; std::vector<float> m_activation_beta; std::vector<std::string> m_activations; float m_clip; std::string m_direction; int m_hidden_size; int m_linear_before_reset;
        std::string m_X_i; std::string m_W_i; std::string m_R_i;
        std::string m_B_i; std::string m_sequence_lens_i; std::string m_initial_h_i;
        
        std::string m_Y_o; std::string m_Y_h_o;

        binding_descriptor   binding;
       

    public:
        GRU(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size,  int _linear_before_reset); 
        virtual void bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _Y_o, std::string _Y_h_o); 
        virtual void build();

        ~GRU() {}
    };
   
}
#endif

