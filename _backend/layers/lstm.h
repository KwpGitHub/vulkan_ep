#ifndef LSTM_H
#define LSTM_H 

#include "../layer.h"

/*

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

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

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: The input sequences packed (and potentially padded) into one 3-D tensor with the shape of `[seq_length, batch_size, input_size]`.
input: The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
input: The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
input: The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not specified - assumed to be 0.
input: Optional tensor specifying lengths of the sequences in a batch. If not specified - assumed all sequences in the batch to have length `seq_length`. It has shape `[batch_size]`.
input: Optional initial value of the hidden. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`.
input: Optional initial value of the cell. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`.
input: The weight tensor for peepholes. Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
output: A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. 
output: The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`.
output: The last output value of the cell. It has shape `[num_directions, batch_size, hidden_size]`.
*/

//LSTM
//INPUTS:                   X_i, W_i, R_i
//OPTIONAL_INPUTS:          B_i, sequence_lens_i, initial_h_i, initial_c_i, P_i
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_o, Y_h_o, Y_c_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>, std::vector<std::string>, float, std::string, int, int


//class stuff
namespace layers {   

    class LSTM : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_activation_alpha; std::vector<float> m_activation_beta; std::vector<std::string> m_activations; float m_clip; std::string m_direction; int m_hidden_size; int m_input_forget;
        std::string m_X_i; std::string m_W_i; std::string m_R_i;
        std::string m_B_i; std::string m_sequence_lens_i; std::string m_initial_h_i; std::string m_initial_c_i; std::string m_P_i;
        
        std::string m_Y_o; std::string m_Y_h_o; std::string m_Y_c_o;

        binding_descriptor   binding;
       

    public:
        LSTM(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size,  int _input_forget); 
        virtual void bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _initial_c_i, std::string _P_i, std::string _Y_o, std::string _Y_h_o, std::string _Y_c_o); 
        virtual void build();

        ~LSTM() {}
    };
   
}
#endif

