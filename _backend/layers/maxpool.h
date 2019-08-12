#ifndef MAXPOOL_H
#define MAXPOOL_H //MaxPool

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         Indices_output_o
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, dilations, pads, storage_order, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t, int, Shape_t



namespace backend {
    class MaxPool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
			
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            Shape_t Indices_output_o;
        };

        vuh::Program<Specs, Params>* program;

    public:
        MaxPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
		
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        std::string Indices_output_o;
        //std::vector<uint32_t> output_shape();
   
        ~MaxPool(){}
    };
}


namespace backend {    
    MaxPool::MaxPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/maxpool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({kernel_shape, auto_pad, ceil_mode, dilations, pads, storage_order, strides, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape(), tensor_dict[Indices_output_o]->shape()}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_output], tensor_dict[Indices_output_o] );
    }

    vuh::Device* MaxPool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
