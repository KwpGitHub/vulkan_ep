#ifndef RNN_H
#define RNN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

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

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: The input sequences packed (and potentially padded) into one 3-D tensor with the shape of `[seq_length, batch_size, input_size]`.
input: The weight tensor for input gate. Concatenation of `Wi` and `WBi` (if bidirectional). The tensor has shape `[num_directions, hidden_size, input_size]`.
input: The recurrence weight tensor. Concatenation of `Ri` and `RBi` (if bidirectional). The tensor has shape `[num_directions, hidden_size, hidden_size]`.
input: The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` and `[WBbi, RBbi]` (if bidirectional). The tensor has shape `[num_directions, 2*hidden_size]`. Optional: If not specified - assumed to be 0.
input: Optional tensor specifying lengths of the sequences in a batch. If not specified - assumed all sequences in the batch to have length `seq_length`. It has shape `[batch_size]`.
input: Optional initial value of the hidden. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`.
output: A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. 
output: The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`.
//*/
//RNN
//INPUTS:                   X_input, W_input, R_input
//OPTIONAL_INPUTS:          B_input_opt, sequence_lens_input_opt, initial_h_input_opt
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_output_opt, Y_h_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class RNN : public Layer {
        typedef struct {
            float clip; int direction; int hidden_size;
			Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
            Shape_t X_input; Shape_t W_input; Shape_t R_input;
            Shape_t B_input_opt; Shape_t sequence_lens_input_opt; Shape_t initial_h_input_opt;
            
            Shape_t Y_output_opt; Shape_t Y_h_output_opt;
        } binding_descriptor;

        float clip; int direction; int hidden_size; std::string activation_alpha; std::string activation_beta; std::string activations;
        std::string X_input; std::string W_input; std::string R_input;
        std::string B_input_opt; std::string sequence_lens_input_opt; std::string initial_h_input_opt;
        
        std::string Y_output_opt; std::string Y_h_output_opt;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RNN(std::string n, float clip, int direction, int hidden_size);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string activation_alpha, std::string activation_beta, std::string activations, std::string X_input, std::string W_input, std::string R_input, std::string B_input_opt, std::string sequence_lens_input_opt, std::string initial_h_input_opt, std::string Y_output_opt, std::string Y_h_output_opt); 

        ~RNN() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    RNN::RNN(std::string n, float clip, int direction, int hidden_size) : Layer(n) { }
       
    vuh::Device* RNN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RNN::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.R_input = tensor_dict[R_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
  		binding.sequence_lens_input_opt = tensor_dict[sequence_lens_input_opt]->shape();
  		binding.initial_h_input_opt = tensor_dict[initial_h_input_opt]->shape();
 
		binding.Y_output_opt = tensor_dict[Y_output_opt]->shape();
  		binding.Y_h_output_opt = tensor_dict[Y_h_output_opt]->shape();
 
		binding.clip = clip;
  		binding.direction = direction;
  		binding.hidden_size = hidden_size;
  		binding.activation_alpha = tensor_dict[activation_alpha]->shape();
  		binding.activation_beta = tensor_dict[activation_beta]->shape();
  		binding.activations = tensor_dict[activations]->shape();
 
    }
    
    void RNN::call(std::string activation_alpha, std::string activation_beta, std::string activations, std::string X_input, std::string W_input, std::string R_input, std::string B_input_opt, std::string sequence_lens_input_opt, std::string initial_h_input_opt, std::string Y_output_opt, std::string Y_h_output_opt){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/rnn.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[activation_alpha]->data(), *tensor_dict[activation_beta]->data(), *tensor_dict[activations]->data(), *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[R_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[sequence_lens_input_opt]->data(), *tensor_dict[initial_h_input_opt]->data(), *tensor_dict[Y_output_opt]->data(), *tensor_dict[Y_h_output_opt]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<RNN, Layer>(m, "RNN")
            .def(py::init<std::string, float, int, int> ())
            .def("forward", &RNN::forward)
            .def("init", &RNN::init)
            .def("call", (void (RNN::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &RNN::call);
    }
}

#endif

/* PYTHON STUFF

*/

