#ifndef LSTM_H
#define LSTM_H //LSTM

//INPUTS:                   X_input, W_input, R_input
//OPTIONAL_INPUTS:          B_input_o, sequence_lens_input_o, initial_h_input_o, initial_c_input_o, P_input_o
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_output_o, Y_h_output_o, Y_c_output_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int, int



namespace backend {
    class LSTM : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float clip; int direction; int hidden_size; int input_forget;
			
            //input
            Shape_t X_input; Shape_t W_input; Shape_t R_input;
            Shape_t B_input_o; Shape_t sequence_lens_input_o; Shape_t initial_h_input_o; Shape_t initial_c_input_o; Shape_t P_input_o;
            //output
            
            Shape_t Y_output_o; Shape_t Y_h_output_o; Shape_t Y_c_output_o;
        };

        vuh::Program<Specs, Params>* program;

    public:
        LSTM(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* activation_alpha; Tensor* activation_beta; Tensor* activations; float clip; int direction; int hidden_size; int input_forget;
		
        //input
        std::string X_input; std::string W_input; std::string R_input;
        std::string B_input_o; std::string sequence_lens_input_o; std::string initial_h_input_o; std::string initial_c_input_o; std::string P_input_o;
        //output
        
        std::string Y_output_o; std::string Y_h_output_o; std::string Y_c_output_o;
        //std::vector<uint32_t> output_shape();
   
        ~LSTM(){}
    };
}


namespace backend {    
    LSTM::LSTM(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/lstm.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({clip, direction, hidden_size, input_forget, tensor_dict[X_input]->shape(), tensor_dict[W_input]->shape(), tensor_dict[R_input]->shape(), tensor_dict[B_input_o]->shape(), tensor_dict[sequence_lens_input_o]->shape(), tensor_dict[initial_h_input_o]->shape(), tensor_dict[initial_c_input_o]->shape(), tensor_dict[P_input_o]->shape(), tensor_dict[Y_output_o]->shape(), tensor_dict[Y_h_output_o]->shape(), tensor_dict[Y_c_output_o]->shape()}, 
                            tensor_dict[X_input], tensor_dict[W_input], tensor_dict[R_input], tensor_dict[B_input_o], tensor_dict[sequence_lens_input_o], tensor_dict[initial_h_input_o], tensor_dict[initial_c_input_o], tensor_dict[P_input_o],
                            tensor_dict[Y_output_o], tensor_dict[Y_h_output_o], tensor_dict[Y_c_output_o] );
    }

    vuh::Device* LSTM::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
