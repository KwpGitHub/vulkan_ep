#ifndef SCATTER_H
#define SCATTER_H //Scatter

//INPUTS:                   data, indices, updates
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class Scatter : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis;
			
            //input
            Shape_t data; Shape_t indices; Shape_t updates;
            
            //output
            Shape_t output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Scatter(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int axis;
		
        //input
        std::string data; std::string indices; std::string updates;
        
        //output
        std::string output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Scatter(){}
    };
}


namespace backend {    
    Scatter::Scatter(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/scatter.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Scatter::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
