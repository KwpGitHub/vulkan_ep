#ifndef LOOP_H
#define LOOP_H //Loop

//INPUTS:                   
//OPTIONAL_INPUTS:          M_output, cond_output
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Loop : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int body;
			
            //input
            
            Shape_t M_output; Shape_t cond_output;
            //output
            
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Loop(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int body;
		
        //input
        
        std::string M_output; std::string cond_output;
        //output
        
        
        //std::vector<uint32_t> output_shape();
   
        ~Loop(){}
    };
}


namespace backend {    
    Loop::Loop(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/loop.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({body}, 
                            tensor_dict[M_output], tensor_dict[cond_output],
                             );
    }

    vuh::Device* Loop::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
