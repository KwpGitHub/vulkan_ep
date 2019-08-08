#ifndef MOD_H
#define MOD_H //Mod

//INPUTS:                   A, B
//OPTIONAL_INPUTS:          
//OUTPUS:                   C
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      fmod
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class Mod : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int fmod;
			
            //input
            Shape_t A; Shape_t B;
            
            //output
            Shape_t C;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Mod(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int fmod;
		
        //input
        std::string A; std::string B;
        
        //output
        std::string C;
        
        //std::vector<uint32_t> output_shape();
   
        ~Mod(){}
    };
}


namespace backend {    
    Mod::Mod(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/mod.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Mod::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
