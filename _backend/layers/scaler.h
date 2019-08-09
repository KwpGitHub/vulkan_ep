#ifndef SCALER_H
#define SCALER_H //Scaler

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*



namespace backend {
    class Scaler : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			Shape_t offset; Shape_t scale;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Scaler(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* offset; Tensor* scale;
		Shape_t offset_t; Shape_t scale_t;
        //input
        std::string X_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~Scaler(){}
    };
}


namespace backend {    
    Scaler::Scaler(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/scaler.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({offset_t, scale_t}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* Scaler::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
