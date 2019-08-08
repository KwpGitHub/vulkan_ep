#ifndef LSTM_H
#define LSTM_H //LSTM

//INPUTS:                   X, W, R
//OPTIONAL_INPUTS:          B, sequence_lens, initial_h, initial_c, P
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y, Y_h, Y_c
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int, int



namespace backend {
    class LSTM : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float clip; int direction; int hidden_size; int input_forget;
			Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
            //input
            Shape_t X; Shape_t W; Shape_t R;
            Shape_t B; Shape_t sequence_lens; Shape_t initial_h; Shape_t initial_c; Shape_t P;
            //output
            
            Shape_t Y; Shape_t Y_h; Shape_t Y_c;
        };

        vuh::Program<Specs, Params>* program;

    public:
        LSTM(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* activation_alpha; Tensor* activation_beta; Tensor* activations; float clip; int direction; int hidden_size; int input_forget;
		Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
        //input
        std::string X; std::string W; std::string R;
        std::string B; std::string sequence_lens; std::string initial_h; std::string initial_c; std::string P;
        //output
        
        std::string Y; std::string Y_h; std::string Y_c;
        //std::vector<uint32_t> output_shape();
   
        ~LSTM(){}
    };
}


namespace backend {    
    LSTM::LSTM(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/lstm.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* LSTM::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
