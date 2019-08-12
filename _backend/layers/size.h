#ifndef SIZE_H
#define SIZE_H //Size

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Size : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t data_input;
            
            //output
            Shape_t size_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Size(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string data_input;
        
        //output
        std::string size_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Size(){}
    };
}


namespace backend {    
    Size::Size(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/size.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({tensor_dict[data_input]->shape(), tensor_dict[size_output]->shape()}, 
                            tensor_dict[data_input],
                            tensor_dict[size_output] );
    }

    vuh::Device* Size::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
