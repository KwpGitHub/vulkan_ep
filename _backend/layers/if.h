#ifndef IF_H
#define IF_H //If

//INPUTS:                   cond_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class If : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int else_branch; int then_branch;
			
            //input
            Shape_t cond_input;
            
            //output
            
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        If(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int else_branch; int then_branch;
		
        //input
        std::string cond_input;
        
        //output
        
        
        //std::vector<uint32_t> output_shape();
   
        ~If(){}
    };
}


namespace backend {    
    If::If(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/if.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({else_branch, then_branch}, 
                            tensor_dict[cond_input],
                             );
    }

    vuh::Device* If::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
