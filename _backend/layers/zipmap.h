#ifndef ZIPMAP_H
#define ZIPMAP_H //ZipMap

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*



namespace backend {
    class ZipMap : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t classlabels_int64s;
			Shape_t classlabels_strings;
            //input
            Shape_t X;
            
            //output
            Shape_t Z;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ZipMap(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t classlabels_int64s; Tensor* classlabels_strings;
		Shape_t classlabels_strings;
        //input
        std::string X;
        
        //output
        std::string Z;
        
        //std::vector<uint32_t> output_shape();
   
        ~ZipMap(){}
    };
}


namespace backend {    
    ZipMap::ZipMap(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/zipmap.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ZipMap::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
