#ifndef SQUEEZE_H
#define SQUEEZE_H //Squeeze

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t



namespace backend {
    class Squeeze : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t axes;
			
            //input
            Shape_t data_input;
            
            //output
            Shape_t squeezed_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Squeeze(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t axes;
		
        //input
        std::string data_input;
        
        //output
        std::string squeezed_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~Squeeze(){}
    };
}


namespace backend {    
    Squeeze::Squeeze(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/squeeze.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({axes}, 
                            tensor_dict[data_input],
                            tensor_dict[squeezed_input_o] );
    }

    vuh::Device* Squeeze::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
