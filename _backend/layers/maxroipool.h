#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H //MaxRoiPool

//INPUTS:                   X_input, rois_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pooled_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      spatial_scale
//OPTIONAL_PARAMETERS_TYPE: float



namespace backend {
    class MaxRoiPool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t pooled_shape; float spatial_scale;
			
            //input
            Shape_t X_input; Shape_t rois_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        MaxRoiPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t pooled_shape; float spatial_scale;
		
        //input
        std::string X_input; std::string rois_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~MaxRoiPool(){}
    };
}


namespace backend {    
    MaxRoiPool::MaxRoiPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/maxroipool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({pooled_shape, spatial_scale}, 
                            tensor_dict[X_input], tensor_dict[rois_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* MaxRoiPool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
