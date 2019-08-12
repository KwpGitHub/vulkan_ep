#ifndef DROPOUT_H
#define DROPOUT_H //Dropout

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         mask_output_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      ratio
//OPTIONAL_PARAMETERS_TYPE: float



namespace backend {
    class Dropout : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float ratio;
			
            //input
            Shape_t data_input;
            
            //output
            Shape_t output_output;
            Shape_t mask_output_o;
        };

        vuh::Program<Specs, Params>* program;

    public:
        Dropout(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float ratio;
		
        //input
        std::string data_input;
        
        //output
        std::string output_output;
        std::string mask_output_o;
        //std::vector<uint32_t> output_shape();
   
        ~Dropout(){}
    };
}


namespace backend {    
    Dropout::Dropout(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/dropout.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({ratio, tensor_dict[data_input]->shape(), tensor_dict[output_output]->shape(), tensor_dict[mask_output_o]->shape()}, 
                            tensor_dict[data_input],
                            tensor_dict[output_output], tensor_dict[mask_output_o] );
    }

    vuh::Device* Dropout::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
