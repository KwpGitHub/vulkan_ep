#ifndef RNN_H
#define RNN_H //RNN

#include "../layer.h"

//INPUTS:                   X_input, W_input, R_input
//OPTIONAL_INPUTS:          B_input_o, sequence_lens_input_o, initial_h_input_o
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_output_o, Y_h_output_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int



namespace backend {
    class RNN : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float clip; int direction; int hidden_size;
			Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
            //input
            Shape_t X_input; Shape_t W_input; Shape_t R_input;
            Shape_t B_input_o; Shape_t sequence_lens_input_o; Shape_t initial_h_input_o;
            //output
            
            Shape_t Y_output_o; Shape_t Y_h_output_o;
        };

        vuh::Program<Specs, Params>* program;

    public:
        RNN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Tensor* activation_alpha; Tensor* activation_beta; Tensor* activations; float clip; int direction; int hidden_size;
		Shape_t activation_alpha_s; Shape_t activation_beta_s; Shape_t activations_s;
        //input
        std::string X_input; std::string W_input; std::string R_input;
        std::string B_input_o; std::string sequence_lens_input_o; std::string initial_h_input_o;
        //output
        
        std::string Y_output_o; std::string Y_h_output_o;
        //std::vector<uint32_t> output_shape();
   
        ~RNN() {}
    };
}


namespace backend {    
    RNN::RNN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/rnn.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({clip, direction, hidden_size, activation_alpha_s, activation_beta_s, activations_s, tensor_dict[X_input]->shape(), tensor_dict[W_input]->shape(), tensor_dict[R_input]->shape(), tensor_dict[B_input_o]->shape(), tensor_dict[sequence_lens_input_o]->shape(), tensor_dict[initial_h_input_o]->shape(), tensor_dict[Y_output_o]->shape(), tensor_dict[Y_h_output_o]->shape()} 
                        , *activation_alpha, *activation_beta, *activations
                        , tensor_dict[X_input], tensor_dict[W_input], tensor_dict[R_input], tensor_dict[B_input_o], tensor_dict[sequence_lens_input_o], tensor_dict[initial_h_input_o], tensor_dict[Y_output_o], tensor_dict[Y_h_output_o] );
    }

    vuh::Device* RNN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
