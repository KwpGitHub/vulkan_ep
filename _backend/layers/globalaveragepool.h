#ifndef GLOBALAVERAGEPOOL_H
#define GLOBALAVERAGEPOOL_H //GlobalAveragePool

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class GlobalAveragePool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        GlobalAveragePool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string X_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~GlobalAveragePool(){}
    };
}


namespace backend {    
    GlobalAveragePool::GlobalAveragePool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/globalaveragepool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* GlobalAveragePool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
