#ifndef LPPOOL_H
#define LPPOOL_H //LpPool

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t



namespace backend {
    class LpPool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t X;
            
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        LpPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
		
        //input
        std::string X;
        
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~LpPool(){}
    };
}


namespace backend {    
    LpPool::LpPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/lppool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* LpPool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
