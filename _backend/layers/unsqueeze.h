#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H //Unsqueeze

//INPUTS:                   data
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Unsqueeze : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t axes;
			
            //input
            Shape_t data;
            
            //output
            Shape_t expanded;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Unsqueeze(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t axes;
		
        //input
        std::string data;
        
        //output
        std::string expanded;
        
        //std::vector<uint32_t> output_shape();
   
        ~Unsqueeze(){}
    };
}


namespace backend {    
    Unsqueeze::Unsqueeze(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/unsqueeze.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Unsqueeze::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
