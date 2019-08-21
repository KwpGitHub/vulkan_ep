void init_layer_LSTM(py::module&);
#include "./layers/lstm.h"
void init_layer_LSTM(py::module& m){
    m.def("_LSTM", [](py::str name, float _clip , int _direction , int _hidden_size , int _input_forget , py::str _activation_alpha , py::str _activation_beta , py::str _activations , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _initial_c_i , py::str _P_i , py::str _Y_o , py::str _Y_h_o , py::str _Y_c_o) {
        auto layer = backend::createInstance<backend::LSTM>(std::string(name));
        //layer->init(_clip, _direction, _hidden_size, _input_forget);    
        //layer->bind(_activation_alpha, _activation_beta, _activations, _X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _initial_c_i, _P_i, _Y_o, _Y_h_o, _Y_c_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Identity(py::module&);
#include "./layers/identity.h"
void init_layer_Identity(py::module& m){
    m.def("_Identity", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Identity>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Abs(py::module&);
#include "./layers/abs.h"
void init_layer_Abs(py::module& m){
    m.def("_Abs", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Abs>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_BatchNormalization(py::module&);
#include "./layers/batchnormalization.h"
void init_layer_BatchNormalization(py::module& m){
    m.def("_BatchNormalization", [](py::str name, float _epsilon , float _momentum , py::str _X_i , py::str _scale_i , py::str _B_i , py::str _mean_i , py::str _var_i , py::str _Y_o , py::str _mean_o , py::str _var_o , py::str _saved_mean_o , py::str _saved_var_o) {
        auto layer = backend::createInstance<backend::BatchNormalization>(std::string(name));
        //layer->init(_epsilon, _momentum);    
        //layer->bind(_X_i, _scale_i, _B_i, _mean_i, _var_i, _Y_o, _mean_o, _var_o, _saved_mean_o, _saved_var_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Mean(py::module&);
#include "./layers/mean.h"
void init_layer_Mean(py::module& m){
    m.def("_Mean", [](py::str name, py::str _mean_o) {
        auto layer = backend::createInstance<backend::Mean>(std::string(name));
        //layer->init();    
        //layer->bind(_mean_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Add(py::module&);
#include "./layers/add.h"
void init_layer_Add(py::module& m){
    m.def("_Add", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Add>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_GlobalMaxPool(py::module&);
#include "./layers/globalmaxpool.h"
void init_layer_GlobalMaxPool(py::module& m){
    m.def("_GlobalMaxPool", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::GlobalMaxPool>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Cast(py::module&);
#include "./layers/cast.h"
void init_layer_Cast(py::module& m){
    m.def("_Cast", [](py::str name, int _to , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Cast>(std::string(name));
        //layer->init(_to);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_AveragePool(py::module&);
#include "./layers/averagepool.h"
void init_layer_AveragePool(py::module& m){
    m.def("_AveragePool", [](py::str name, py::list _kernel_shape , int _auto_pad , int _ceil_mode , int _count_include_pad , py::list _pads , py::list _strides , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::AveragePool>(std::string(name));
        //layer->init(_kernel_shape, _auto_pad, _ceil_mode, _count_include_pad, _pads, _strides);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_And(py::module&);
#include "./layers/and.h"
void init_layer_And(py::module& m){
    m.def("_And", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::And>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LRN(py::module&);
#include "./layers/lrn.h"
void init_layer_LRN(py::module& m){
    m.def("_LRN", [](py::str name, int _size , float _alpha , float _beta , float _bias , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::LRN>(std::string(name));
        //layer->init(_size, _alpha, _beta, _bias);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ArgMax(py::module&);
#include "./layers/argmax.h"
void init_layer_ArgMax(py::module& m){
    m.def("_ArgMax", [](py::str name, int _axis , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ArgMax>(std::string(name));
        //layer->init(_axis, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Resize(py::module&);
#include "./layers/resize.h"
void init_layer_Resize(py::module& m){
    m.def("_Resize", [](py::str name, int _mode , py::str _X_i , py::str _scales_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Resize>(std::string(name));
        //layer->init(_mode);    
        //layer->bind(_X_i, _scales_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Expand(py::module&);
#include "./layers/expand.h"
void init_layer_Expand(py::module& m){
    m.def("_Expand", [](py::str name, py::str _input_i , py::str _shape_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Expand>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _shape_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Neg(py::module&);
#include "./layers/neg.h"
void init_layer_Neg(py::module& m){
    m.def("_Neg", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Neg>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Mul(py::module&);
#include "./layers/mul.h"
void init_layer_Mul(py::module& m){
    m.def("_Mul", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Mul>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ArgMin(py::module&);
#include "./layers/argmin.h"
void init_layer_ArgMin(py::module& m){
    m.def("_ArgMin", [](py::str name, int _axis , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ArgMin>(std::string(name));
        //layer->init(_axis, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_CastMap(py::module&);
#include "./layers/castmap.h"
void init_layer_CastMap(py::module& m){
    m.def("_CastMap", [](py::str name, int _cast_to , int _map_form , int _max_map , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::CastMap>(std::string(name));
        //layer->init(_cast_to, _map_form, _max_map);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Exp(py::module&);
#include "./layers/exp.h"
void init_layer_Exp(py::module& m){
    m.def("_Exp", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Exp>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Div(py::module&);
#include "./layers/div.h"
void init_layer_Div(py::module& m){
    m.def("_Div", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Div>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReverseSequence(py::module&);
#include "./layers/reversesequence.h"
void init_layer_ReverseSequence(py::module& m){
    m.def("_ReverseSequence", [](py::str name, int _batch_axis , int _time_axis , py::str _input_i , py::str _sequence_lens_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::ReverseSequence>(std::string(name));
        //layer->init(_batch_axis, _time_axis);    
        //layer->bind(_input_i, _sequence_lens_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Ceil(py::module&);
#include "./layers/ceil.h"
void init_layer_Ceil(py::module& m){
    m.def("_Ceil", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Ceil>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_DepthToSpace(py::module&);
#include "./layers/depthtospace.h"
void init_layer_DepthToSpace(py::module& m){
    m.def("_DepthToSpace", [](py::str name, int _blocksize , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::DepthToSpace>(std::string(name));
        //layer->init(_blocksize);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Clip(py::module&);
#include "./layers/clip.h"
void init_layer_Clip(py::module& m){
    m.def("_Clip", [](py::str name, float _max , float _min , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Clip>(std::string(name));
        //layer->init(_max, _min);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RNN(py::module&);
#include "./layers/rnn.h"
void init_layer_RNN(py::module& m){
    m.def("_RNN", [](py::str name, float _clip , int _direction , int _hidden_size , py::str _activation_alpha , py::str _activation_beta , py::str _activations , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _Y_o , py::str _Y_h_o) {
        auto layer = backend::createInstance<backend::RNN>(std::string(name));
        //layer->init(_clip, _direction, _hidden_size);    
        //layer->bind(_activation_alpha, _activation_beta, _activations, _X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _Y_o, _Y_h_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Concat(py::module&);
#include "./layers/concat.h"
void init_layer_Concat(py::module& m){
    m.def("_Concat", [](py::str name, int _axis , py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _concat_result_o) {
        auto layer = backend::createInstance<backend::Concat>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _concat_result_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Constant(py::module&);
#include "./layers/constant.h"
void init_layer_Constant(py::module& m){
    m.def("_Constant", [](py::str name, py::str _value , py::str _output_o) {
        auto layer = backend::createInstance<backend::Constant>(std::string(name));
        //layer->init();    
        //layer->bind(_value, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LpPool(py::module&);
#include "./layers/lppool.h"
void init_layer_LpPool(py::module& m){
    m.def("_LpPool", [](py::str name, py::list _kernel_shape , int _auto_pad , int _p , py::list _pads , py::list _strides , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::LpPool>(std::string(name));
        //layer->init(_kernel_shape, _auto_pad, _p, _pads, _strides);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Conv(py::module&);
#include "./layers/conv.h"
void init_layer_Conv(py::module& m){
    m.def("_Conv", [](py::str name, int _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _W_i , py::str _B_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Conv>(std::string(name));
        //layer->init(_auto_pad, _dilations, _group, _kernel_shape, _pads, _strides);    
        //layer->bind(_X_i, _W_i, _B_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Not(py::module&);
#include "./layers/not.h"
void init_layer_Not(py::module& m){
    m.def("_Not", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Not>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Gather(py::module&);
#include "./layers/gather.h"
void init_layer_Gather(py::module& m){
    m.def("_Gather", [](py::str name, int _axis , py::str _data_i , py::str _indices_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Gather>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_data_i, _indices_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ConvTranspose(py::module&);
#include "./layers/convtranspose.h"
void init_layer_ConvTranspose(py::module& m){
    m.def("_ConvTranspose", [](py::str name, int _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _output_padding , py::list _output_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _W_i , py::str _B_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::ConvTranspose>(std::string(name));
        //layer->init(_auto_pad, _dilations, _group, _kernel_shape, _output_padding, _output_shape, _pads, _strides);    
        //layer->bind(_X_i, _W_i, _B_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Dropout(py::module&);
#include "./layers/dropout.h"
void init_layer_Dropout(py::module& m){
    m.def("_Dropout", [](py::str name, float _ratio , py::str _data_i , py::str _output_o , py::str _mask_o) {
        auto layer = backend::createInstance<backend::Dropout>(std::string(name));
        //layer->init(_ratio);    
        //layer->bind(_data_i, _output_o, _mask_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LeakyRelu(py::module&);
#include "./layers/leakyrelu.h"
void init_layer_LeakyRelu(py::module& m){
    m.def("_LeakyRelu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::LeakyRelu>(std::string(name));
        //layer->init(_alpha);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Elu(py::module&);
#include "./layers/elu.h"
void init_layer_Elu(py::module& m){
    m.def("_Elu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Elu>(std::string(name));
        //layer->init(_alpha);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_GlobalAveragePool(py::module&);
#include "./layers/globalaveragepool.h"
void init_layer_GlobalAveragePool(py::module& m){
    m.def("_GlobalAveragePool", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::GlobalAveragePool>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Gemm(py::module&);
#include "./layers/gemm.h"
void init_layer_Gemm(py::module& m){
    m.def("_Gemm", [](py::str name, float _alpha , float _beta , int _transA , int _transB , py::str _A_i , py::str _B_i , py::str _C_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Gemm>(std::string(name));
        //layer->init(_alpha, _beta, _transA, _transB);    
        //layer->bind(_A_i, _B_i, _C_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MaxPool(py::module&);
#include "./layers/maxpool.h"
void init_layer_MaxPool(py::module& m){
    m.def("_MaxPool", [](py::str name, py::list _kernel_shape , int _auto_pad , int _ceil_mode , py::list _dilations , py::list _pads , int _storage_order , py::list _strides , py::str _X_i , py::str _Y_o , py::str _Indices_o) {
        auto layer = backend::createInstance<backend::MaxPool>(std::string(name));
        //layer->init(_kernel_shape, _auto_pad, _ceil_mode, _dilations, _pads, _storage_order, _strides);    
        //layer->bind(_X_i, _Y_o, _Indices_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Equal(py::module&);
#include "./layers/equal.h"
void init_layer_Equal(py::module& m){
    m.def("_Equal", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Equal>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Tile(py::module&);
#include "./layers/tile.h"
void init_layer_Tile(py::module& m){
    m.def("_Tile", [](py::str name, py::str _input_i , py::str _repeats_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Tile>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _repeats_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Flatten(py::module&);
#include "./layers/flatten.h"
void init_layer_Flatten(py::module& m){
    m.def("_Flatten", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Flatten>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Floor(py::module&);
#include "./layers/floor.h"
void init_layer_Floor(py::module& m){
    m.def("_Floor", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Floor>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_GRU(py::module&);
#include "./layers/gru.h"
void init_layer_GRU(py::module& m){
    m.def("_GRU", [](py::str name, float _clip , int _direction , int _hidden_size , int _linear_before_reset , py::str _activation_alpha , py::str _activation_beta , py::str _activations , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _Y_o , py::str _Y_h_o) {
        auto layer = backend::createInstance<backend::GRU>(std::string(name));
        //layer->init(_clip, _direction, _hidden_size, _linear_before_reset);    
        //layer->bind(_activation_alpha, _activation_beta, _activations, _X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _Y_o, _Y_h_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_GlobalLpPool(py::module&);
#include "./layers/globallppool.h"
void init_layer_GlobalLpPool(py::module& m){
    m.def("_GlobalLpPool", [](py::str name, int _p , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::GlobalLpPool>(std::string(name));
        //layer->init(_p);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Greater(py::module&);
#include "./layers/greater.h"
void init_layer_Greater(py::module& m){
    m.def("_Greater", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Greater>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_HardSigmoid(py::module&);
#include "./layers/hardsigmoid.h"
void init_layer_HardSigmoid(py::module& m){
    m.def("_HardSigmoid", [](py::str name, float _alpha , float _beta , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::HardSigmoid>(std::string(name));
        //layer->init(_alpha, _beta);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Selu(py::module&);
#include "./layers/selu.h"
void init_layer_Selu(py::module& m){
    m.def("_Selu", [](py::str name, float _alpha , float _gamma , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Selu>(std::string(name));
        //layer->init(_alpha, _gamma);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Hardmax(py::module&);
#include "./layers/hardmax.h"
void init_layer_Hardmax(py::module& m){
    m.def("_Hardmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Hardmax>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_If(py::module&);
#include "./layers/if.h"
void init_layer_If(py::module& m){
    m.def("_If", [](py::str name, int _else_branch , int _then_branch , py::str _cond_i) {
        auto layer = backend::createInstance<backend::If>(std::string(name));
        //layer->init(_else_branch, _then_branch);    
        //layer->bind(_cond_i); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Min(py::module&);
#include "./layers/min.h"
void init_layer_Min(py::module& m){
    m.def("_Min", [](py::str name, py::str _min_o) {
        auto layer = backend::createInstance<backend::Min>(std::string(name));
        //layer->init();    
        //layer->bind(_min_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_InstanceNormalization(py::module&);
#include "./layers/instancenormalization.h"
void init_layer_InstanceNormalization(py::module& m){
    m.def("_InstanceNormalization", [](py::str name, float _epsilon , py::str _input_i , py::str _scale_i , py::str _B_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::InstanceNormalization>(std::string(name));
        //layer->init(_epsilon);    
        //layer->bind(_input_i, _scale_i, _B_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Less(py::module&);
#include "./layers/less.h"
void init_layer_Less(py::module& m){
    m.def("_Less", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Less>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_EyeLike(py::module&);
#include "./layers/eyelike.h"
void init_layer_EyeLike(py::module& m){
    m.def("_EyeLike", [](py::str name, int _dtype , int _k , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::EyeLike>(std::string(name));
        //layer->init(_dtype, _k);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RandomNormal(py::module&);
#include "./layers/randomnormal.h"
void init_layer_RandomNormal(py::module& m){
    m.def("_RandomNormal", [](py::str name, py::list _shape , int _dtype , float _mean , float _scale , float _seed , py::str _output_o) {
        auto layer = backend::createInstance<backend::RandomNormal>(std::string(name));
        //layer->init(_shape, _dtype, _mean, _scale, _seed);    
        //layer->bind(_output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Slice(py::module&);
#include "./layers/slice.h"
void init_layer_Slice(py::module& m){
    m.def("_Slice", [](py::str name, py::str _data_i , py::str _starts_i , py::str _ends_i , py::str _axes_i , py::str _steps_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Slice>(std::string(name));
        //layer->init();    
        //layer->bind(_data_i, _starts_i, _ends_i, _axes_i, _steps_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_PRelu(py::module&);
#include "./layers/prelu.h"
void init_layer_PRelu(py::module& m){
    m.def("_PRelu", [](py::str name, py::str _X_i , py::str _slope_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::PRelu>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _slope_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Log(py::module&);
#include "./layers/log.h"
void init_layer_Log(py::module& m){
    m.def("_Log", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Log>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LogSoftmax(py::module&);
#include "./layers/logsoftmax.h"
void init_layer_LogSoftmax(py::module& m){
    m.def("_LogSoftmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::LogSoftmax>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Loop(py::module&);
#include "./layers/loop.h"
void init_layer_Loop(py::module& m){
    m.def("_Loop", [](py::str name, int _body , py::str _M_i , py::str _cond_i) {
        auto layer = backend::createInstance<backend::Loop>(std::string(name));
        //layer->init(_body);    
        //layer->bind(_M_i, _cond_i); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LpNormalization(py::module&);
#include "./layers/lpnormalization.h"
void init_layer_LpNormalization(py::module& m){
    m.def("_LpNormalization", [](py::str name, int _axis , int _p , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::LpNormalization>(std::string(name));
        //layer->init(_axis, _p);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MatMul(py::module&);
#include "./layers/matmul.h"
void init_layer_MatMul(py::module& m){
    m.def("_MatMul", [](py::str name, py::str _A_i , py::str _B_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::MatMul>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceL2(py::module&);
#include "./layers/reducel2.h"
void init_layer_ReduceL2(py::module& m){
    m.def("_ReduceL2", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceL2>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Max(py::module&);
#include "./layers/max.h"
void init_layer_Max(py::module& m){
    m.def("_Max", [](py::str name, py::str _max_o) {
        auto layer = backend::createInstance<backend::Max>(std::string(name));
        //layer->init();    
        //layer->bind(_max_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MaxRoiPool(py::module&);
#include "./layers/maxroipool.h"
void init_layer_MaxRoiPool(py::module& m){
    m.def("_MaxRoiPool", [](py::str name, py::list _pooled_shape , float _spatial_scale , py::str _X_i , py::str _rois_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::MaxRoiPool>(std::string(name));
        //layer->init(_pooled_shape, _spatial_scale);    
        //layer->bind(_X_i, _rois_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Or(py::module&);
#include "./layers/or.h"
void init_layer_Or(py::module& m){
    m.def("_Or", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Or>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Pad(py::module&);
#include "./layers/pad.h"
void init_layer_Pad(py::module& m){
    m.def("_Pad", [](py::str name, py::list _pads , int _mode , float _value , py::str _data_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Pad>(std::string(name));
        //layer->init(_pads, _mode, _value);    
        //layer->bind(_data_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RandomUniformLike(py::module&);
#include "./layers/randomuniformlike.h"
void init_layer_RandomUniformLike(py::module& m){
    m.def("_RandomUniformLike", [](py::str name, int _dtype , float _high , float _low , float _seed , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::RandomUniformLike>(std::string(name));
        //layer->init(_dtype, _high, _low, _seed);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Reciprocal(py::module&);
#include "./layers/reciprocal.h"
void init_layer_Reciprocal(py::module& m){
    m.def("_Reciprocal", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Reciprocal>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Pow(py::module&);
#include "./layers/pow.h"
void init_layer_Pow(py::module& m){
    m.def("_Pow", [](py::str name, py::str _X_i , py::str _Y_i , py::str _Z_o) {
        auto layer = backend::createInstance<backend::Pow>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_i, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RandomNormalLike(py::module&);
#include "./layers/randomnormallike.h"
void init_layer_RandomNormalLike(py::module& m){
    m.def("_RandomNormalLike", [](py::str name, int _dtype , float _mean , float _scale , float _seed , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::RandomNormalLike>(std::string(name));
        //layer->init(_dtype, _mean, _scale, _seed);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_OneHot(py::module&);
#include "./layers/onehot.h"
void init_layer_OneHot(py::module& m){
    m.def("_OneHot", [](py::str name, int _axis , py::str _indices_i , py::str _depth_i , py::str _values_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::OneHot>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_indices_i, _depth_i, _values_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RandomUniform(py::module&);
#include "./layers/randomuniform.h"
void init_layer_RandomUniform(py::module& m){
    m.def("_RandomUniform", [](py::str name, py::list _shape , int _dtype , float _high , float _low , float _seed , py::str _output_o) {
        auto layer = backend::createInstance<backend::RandomUniform>(std::string(name));
        //layer->init(_shape, _dtype, _high, _low, _seed);    
        //layer->bind(_output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceL1(py::module&);
#include "./layers/reducel1.h"
void init_layer_ReduceL1(py::module& m){
    m.def("_ReduceL1", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceL1>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceLogSum(py::module&);
#include "./layers/reducelogsum.h"
void init_layer_ReduceLogSum(py::module& m){
    m.def("_ReduceLogSum", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceLogSum>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceLogSumExp(py::module&);
#include "./layers/reducelogsumexp.h"
void init_layer_ReduceLogSumExp(py::module& m){
    m.def("_ReduceLogSumExp", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceLogSumExp>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceMax(py::module&);
#include "./layers/reducemax.h"
void init_layer_ReduceMax(py::module& m){
    m.def("_ReduceMax", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceMax>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_OneHotEncoder(py::module&);
#include "./layers/onehotencoder.h"
void init_layer_OneHotEncoder(py::module& m){
    m.def("_OneHotEncoder", [](py::str name, py::list _cats_int64s , int _zeros , py::str _cats_strings , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::OneHotEncoder>(std::string(name));
        //layer->init(_cats_int64s, _zeros);    
        //layer->bind(_cats_strings, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_IsNaN(py::module&);
#include "./layers/isnan.h"
void init_layer_IsNaN(py::module& m){
    m.def("_IsNaN", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::IsNaN>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceMean(py::module&);
#include "./layers/reducemean.h"
void init_layer_ReduceMean(py::module& m){
    m.def("_ReduceMean", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceMean>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceMin(py::module&);
#include "./layers/reducemin.h"
void init_layer_ReduceMin(py::module& m){
    m.def("_ReduceMin", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceMin>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_TreeEnsembleRegressor(py::module&);
#include "./layers/treeensembleregressor.h"
void init_layer_TreeEnsembleRegressor(py::module& m){
    m.def("_TreeEnsembleRegressor", [](py::str name, int _aggregate_function , int _n_targets , py::list _nodes_falsenodeids , py::list _nodes_featureids , py::list _nodes_missing_value_tracks_true , py::list _nodes_nodeids , py::list _nodes_treeids , py::list _nodes_truenodeids , int _post_transform , py::list _target_ids , py::list _target_nodeids , py::list _target_treeids , py::str _base_values , py::str _nodes_hitrates , py::str _nodes_modes , py::str _nodes_values , py::str _target_weights , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::TreeEnsembleRegressor>(std::string(name));
        //layer->init(_aggregate_function, _n_targets, _nodes_falsenodeids, _nodes_featureids, _nodes_missing_value_tracks_true, _nodes_nodeids, _nodes_treeids, _nodes_truenodeids, _post_transform, _target_ids, _target_nodeids, _target_treeids);    
        //layer->bind(_base_values, _nodes_hitrates, _nodes_modes, _nodes_values, _target_weights, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceProd(py::module&);
#include "./layers/reduceprod.h"
void init_layer_ReduceProd(py::module& m){
    m.def("_ReduceProd", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceProd>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceSum(py::module&);
#include "./layers/reducesum.h"
void init_layer_ReduceSum(py::module& m){
    m.def("_ReduceSum", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceSum>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ReduceSumSquare(py::module&);
#include "./layers/reducesumsquare.h"
void init_layer_ReduceSumSquare(py::module& m){
    m.def("_ReduceSumSquare", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        auto layer = backend::createInstance<backend::ReduceSumSquare>(std::string(name));
        //layer->init(_axes, _keepdims);    
        //layer->bind(_data_i, _reduced_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Relu(py::module&);
#include "./layers/relu.h"
void init_layer_Relu(py::module& m){
    m.def("_Relu", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Relu>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Reshape(py::module&);
#include "./layers/reshape.h"
void init_layer_Reshape(py::module& m){
    m.def("_Reshape", [](py::str name, py::str _data_i , py::str _shape_i , py::str _reshaped_o) {
        auto layer = backend::createInstance<backend::Reshape>(std::string(name));
        //layer->init();    
        //layer->bind(_data_i, _shape_i, _reshaped_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Shape(py::module&);
#include "./layers/shape.h"
void init_layer_Shape(py::module& m){
    m.def("_Shape", [](py::str name, py::str _data_i , py::str _shape_o) {
        auto layer = backend::createInstance<backend::Shape>(std::string(name));
        //layer->init();    
        //layer->bind(_data_i, _shape_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sigmoid(py::module&);
#include "./layers/sigmoid.h"
void init_layer_Sigmoid(py::module& m){
    m.def("_Sigmoid", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Sigmoid>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Size(py::module&);
#include "./layers/size.h"
void init_layer_Size(py::module& m){
    m.def("_Size", [](py::str name, py::str _data_i , py::str _size_o) {
        auto layer = backend::createInstance<backend::Size>(std::string(name));
        //layer->init();    
        //layer->bind(_data_i, _size_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Softmax(py::module&);
#include "./layers/softmax.h"
void init_layer_Softmax(py::module& m){
    m.def("_Softmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Softmax>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Softplus(py::module&);
#include "./layers/softplus.h"
void init_layer_Softplus(py::module& m){
    m.def("_Softplus", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Softplus>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Softsign(py::module&);
#include "./layers/softsign.h"
void init_layer_Softsign(py::module& m){
    m.def("_Softsign", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Softsign>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_SpaceToDepth(py::module&);
#include "./layers/spacetodepth.h"
void init_layer_SpaceToDepth(py::module& m){
    m.def("_SpaceToDepth", [](py::str name, int _blocksize , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::SpaceToDepth>(std::string(name));
        //layer->init(_blocksize);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_TfIdfVectorizer(py::module&);
#include "./layers/tfidfvectorizer.h"
void init_layer_TfIdfVectorizer(py::module& m){
    m.def("_TfIdfVectorizer", [](py::str name, int _max_gram_length , int _max_skip_count , int _min_gram_length , int _mode , py::list _ngram_counts , py::list _ngram_indexes , py::list _pool_int64s , py::str _pool_strings , py::str _weights , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::TfIdfVectorizer>(std::string(name));
        //layer->init(_max_gram_length, _max_skip_count, _min_gram_length, _mode, _ngram_counts, _ngram_indexes, _pool_int64s);    
        //layer->bind(_pool_strings, _weights, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Split(py::module&);
#include "./layers/split.h"
void init_layer_Split(py::module& m){
    m.def("_Split", [](py::str name, int _axis , py::list _split , py::str _input_i) {
        auto layer = backend::createInstance<backend::Split>(std::string(name));
        //layer->init(_axis, _split);    
        //layer->bind(_input_i); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Imputer(py::module&);
#include "./layers/imputer.h"
void init_layer_Imputer(py::module& m){
    m.def("_Imputer", [](py::str name, py::list _imputed_value_int64s , float _replaced_value_float , int _replaced_value_int64 , py::str _imputed_value_floats , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Imputer>(std::string(name));
        //layer->init(_imputed_value_int64s, _replaced_value_float, _replaced_value_int64);    
        //layer->bind(_imputed_value_floats, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sqrt(py::module&);
#include "./layers/sqrt.h"
void init_layer_Sqrt(py::module& m){
    m.def("_Sqrt", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Sqrt>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Squeeze(py::module&);
#include "./layers/squeeze.h"
void init_layer_Squeeze(py::module& m){
    m.def("_Squeeze", [](py::str name, py::list _axes , py::str _data_i , py::str _squeezed_o) {
        auto layer = backend::createInstance<backend::Squeeze>(std::string(name));
        //layer->init(_axes);    
        //layer->bind(_data_i, _squeezed_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_TopK(py::module&);
#include "./layers/topk.h"
void init_layer_TopK(py::module& m){
    m.def("_TopK", [](py::str name, int _axis , py::str _X_i , py::str _K_i , py::str _Values_o , py::str _Indices_o) {
        auto layer = backend::createInstance<backend::TopK>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_X_i, _K_i, _Values_o, _Indices_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sub(py::module&);
#include "./layers/sub.h"
void init_layer_Sub(py::module& m){
    m.def("_Sub", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Sub>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sum(py::module&);
#include "./layers/sum.h"
void init_layer_Sum(py::module& m){
    m.def("_Sum", [](py::str name, py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _sum_o) {
        auto layer = backend::createInstance<backend::Sum>(std::string(name));
        //layer->init();    
        //layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _sum_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Shrink(py::module&);
#include "./layers/shrink.h"
void init_layer_Shrink(py::module& m){
    m.def("_Shrink", [](py::str name, float _bias , float _lambd , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Shrink>(std::string(name));
        //layer->init(_bias, _lambd);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Tanh(py::module&);
#include "./layers/tanh.h"
void init_layer_Tanh(py::module& m){
    m.def("_Tanh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Tanh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Transpose(py::module&);
#include "./layers/transpose.h"
void init_layer_Transpose(py::module& m){
    m.def("_Transpose", [](py::str name, py::list _perm , py::str _data_i , py::str _transposed_o) {
        auto layer = backend::createInstance<backend::Transpose>(std::string(name));
        //layer->init(_perm);    
        //layer->bind(_data_i, _transposed_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Unsqueeze(py::module&);
#include "./layers/unsqueeze.h"
void init_layer_Unsqueeze(py::module& m){
    m.def("_Unsqueeze", [](py::str name, py::list _axes , py::str _data_i , py::str _expanded_o) {
        auto layer = backend::createInstance<backend::Unsqueeze>(std::string(name));
        //layer->init(_axes);    
        //layer->bind(_data_i, _expanded_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_SVMClassifier(py::module&);
#include "./layers/svmclassifier.h"
void init_layer_SVMClassifier(py::module& m){
    m.def("_SVMClassifier", [](py::str name, py::list _classlabels_ints , int _kernel_type , int _post_transform , py::list _vectors_per_class , py::str _classlabels_strings , py::str _coefficients , py::str _kernel_params , py::str _prob_a , py::str _prob_b , py::str _rho , py::str _support_vectors , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        auto layer = backend::createInstance<backend::SVMClassifier>(std::string(name));
        //layer->init(_classlabels_ints, _kernel_type, _post_transform, _vectors_per_class);    
        //layer->bind(_classlabels_strings, _coefficients, _kernel_params, _prob_a, _prob_b, _rho, _support_vectors, _X_i, _Y_o, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Xor(py::module&);
#include "./layers/xor.h"
void init_layer_Xor(py::module& m){
    m.def("_Xor", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Xor>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Acos(py::module&);
#include "./layers/acos.h"
void init_layer_Acos(py::module& m){
    m.def("_Acos", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Acos>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Asin(py::module&);
#include "./layers/asin.h"
void init_layer_Asin(py::module& m){
    m.def("_Asin", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Asin>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Atan(py::module&);
#include "./layers/atan.h"
void init_layer_Atan(py::module& m){
    m.def("_Atan", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Atan>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Cos(py::module&);
#include "./layers/cos.h"
void init_layer_Cos(py::module& m){
    m.def("_Cos", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Cos>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sin(py::module&);
#include "./layers/sin.h"
void init_layer_Sin(py::module& m){
    m.def("_Sin", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Sin>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Tan(py::module&);
#include "./layers/tan.h"
void init_layer_Tan(py::module& m){
    m.def("_Tan", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Tan>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Multinomial(py::module&);
#include "./layers/multinomial.h"
void init_layer_Multinomial(py::module& m){
    m.def("_Multinomial", [](py::str name, int _dtype , int _sample_size , float _seed , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Multinomial>(std::string(name));
        //layer->init(_dtype, _sample_size, _seed);    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Scan(py::module&);
#include "./layers/scan.h"
void init_layer_Scan(py::module& m){
    m.def("_Scan", [](py::str name, int _body , int _num_scan_inputs , py::list _scan_input_axes , py::list _scan_input_directions , py::list _scan_output_axes , py::list _scan_output_directions , py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _y0_o , py::str _y1_o , py::str _y2_o , py::str _y3_o , py::str _y4_o , py::str _y5_o , py::str _y6_o , py::str _y7_o , py::str _y8_o , py::str _y9_o , py::str _y10_o , py::str _y11_o , py::str _y12_o , py::str _y13_o , py::str _y14_o , py::str _y15_o , py::str _y16_o , py::str _y17_o , py::str _y18_o , py::str _y19_o , py::str _y20_o , py::str _y21_o , py::str _y22_o , py::str _y23_o , py::str _y24_o , py::str _y25_o , py::str _y26_o , py::str _y27_o , py::str _y28_o , py::str _y29_o , py::str _y30_o , py::str _y31_o) {
        auto layer = backend::createInstance<backend::Scan>(std::string(name));
        //layer->init(_body, _num_scan_inputs, _scan_input_axes, _scan_input_directions, _scan_output_axes, _scan_output_directions);    
        //layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _y0_o, _y1_o, _y2_o, _y3_o, _y4_o, _y5_o, _y6_o, _y7_o, _y8_o, _y9_o, _y10_o, _y11_o, _y12_o, _y13_o, _y14_o, _y15_o, _y16_o, _y17_o, _y18_o, _y19_o, _y20_o, _y21_o, _y22_o, _y23_o, _y24_o, _y25_o, _y26_o, _y27_o, _y28_o, _y29_o, _y30_o, _y31_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Compress(py::module&);
#include "./layers/compress.h"
void init_layer_Compress(py::module& m){
    m.def("_Compress", [](py::str name, int _axis , py::str _input_i , py::str _condition_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Compress>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_input_i, _condition_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ConstantOfShape(py::module&);
#include "./layers/constantofshape.h"
void init_layer_ConstantOfShape(py::module& m){
    m.def("_ConstantOfShape", [](py::str name, py::str _value , py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::ConstantOfShape>(std::string(name));
        //layer->init();    
        //layer->bind(_value, _input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MaxUnpool(py::module&);
#include "./layers/maxunpool.h"
void init_layer_MaxUnpool(py::module& m){
    m.def("_MaxUnpool", [](py::str name, py::list _kernel_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _I_i , py::str _output_shape_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::MaxUnpool>(std::string(name));
        //layer->init(_kernel_shape, _pads, _strides);    
        //layer->bind(_X_i, _I_i, _output_shape_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Scatter(py::module&);
#include "./layers/scatter.h"
void init_layer_Scatter(py::module& m){
    m.def("_Scatter", [](py::str name, int _axis , py::str _data_i , py::str _indices_i , py::str _updates_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Scatter>(std::string(name));
        //layer->init(_axis);    
        //layer->bind(_data_i, _indices_i, _updates_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sinh(py::module&);
#include "./layers/sinh.h"
void init_layer_Sinh(py::module& m){
    m.def("_Sinh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Sinh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Cosh(py::module&);
#include "./layers/cosh.h"
void init_layer_Cosh(py::module& m){
    m.def("_Cosh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Cosh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Asinh(py::module&);
#include "./layers/asinh.h"
void init_layer_Asinh(py::module& m){
    m.def("_Asinh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Asinh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Acosh(py::module&);
#include "./layers/acosh.h"
void init_layer_Acosh(py::module& m){
    m.def("_Acosh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Acosh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_NonMaxSuppression(py::module&);
#include "./layers/nonmaxsuppression.h"
void init_layer_NonMaxSuppression(py::module& m){
    m.def("_NonMaxSuppression", [](py::str name, int _center_point_box , py::str _boxes_i , py::str _scores_i , py::str _max_output_boxes_per_class_i , py::str _iou_threshold_i , py::str _score_threshold_i , py::str _selected_indices_o) {
        auto layer = backend::createInstance<backend::NonMaxSuppression>(std::string(name));
        //layer->init(_center_point_box);    
        //layer->bind(_boxes_i, _scores_i, _max_output_boxes_per_class_i, _iou_threshold_i, _score_threshold_i, _selected_indices_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Atanh(py::module&);
#include "./layers/atanh.h"
void init_layer_Atanh(py::module& m){
    m.def("_Atanh", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Atanh>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Sign(py::module&);
#include "./layers/sign.h"
void init_layer_Sign(py::module& m){
    m.def("_Sign", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Sign>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Erf(py::module&);
#include "./layers/erf.h"
void init_layer_Erf(py::module& m){
    m.def("_Erf", [](py::str name, py::str _input_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Erf>(std::string(name));
        //layer->init();    
        //layer->bind(_input_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Where(py::module&);
#include "./layers/where.h"
void init_layer_Where(py::module& m){
    m.def("_Where", [](py::str name, py::str _condition_i , py::str _X_i , py::str _Y_i , py::str _output_o) {
        auto layer = backend::createInstance<backend::Where>(std::string(name));
        //layer->init();    
        //layer->bind(_condition_i, _X_i, _Y_i, _output_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_NonZero(py::module&);
#include "./layers/nonzero.h"
void init_layer_NonZero(py::module& m){
    m.def("_NonZero", [](py::str name, py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::NonZero>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MeanVarianceNormalization(py::module&);
#include "./layers/meanvariancenormalization.h"
void init_layer_MeanVarianceNormalization(py::module& m){
    m.def("_MeanVarianceNormalization", [](py::str name, py::list _axes , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::MeanVarianceNormalization>(std::string(name));
        //layer->init(_axes);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_StringNormalizer(py::module&);
#include "./layers/stringnormalizer.h"
void init_layer_StringNormalizer(py::module& m){
    m.def("_StringNormalizer", [](py::str name, int _case_change_action , int _is_case_sensitive , int _locale , py::str _stopwords , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::StringNormalizer>(std::string(name));
        //layer->init(_case_change_action, _is_case_sensitive, _locale);    
        //layer->bind(_stopwords, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Mod(py::module&);
#include "./layers/mod.h"
void init_layer_Mod(py::module& m){
    m.def("_Mod", [](py::str name, int _fmod , py::str _A_i , py::str _B_i , py::str _C_o) {
        auto layer = backend::createInstance<backend::Mod>(std::string(name));
        //layer->init(_fmod);    
        //layer->bind(_A_i, _B_i, _C_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ThresholdedRelu(py::module&);
#include "./layers/thresholdedrelu.h"
void init_layer_ThresholdedRelu(py::module& m){
    m.def("_ThresholdedRelu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::ThresholdedRelu>(std::string(name));
        //layer->init(_alpha);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_MatMulInteger(py::module&);
#include "./layers/matmulinteger.h"
void init_layer_MatMulInteger(py::module& m){
    m.def("_MatMulInteger", [](py::str name, py::str _A_i , py::str _B_i , py::str _a_zero_point_i , py::str _b_zero_point_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::MatMulInteger>(std::string(name));
        //layer->init();    
        //layer->bind(_A_i, _B_i, _a_zero_point_i, _b_zero_point_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_QLinearMatMul(py::module&);
#include "./layers/qlinearmatmul.h"
void init_layer_QLinearMatMul(py::module& m){
    m.def("_QLinearMatMul", [](py::str name, py::str _a_i , py::str _a_scale_i , py::str _a_zero_point_i , py::str _b_i , py::str _b_scale_i , py::str _b_zero_point_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _y_o) {
        auto layer = backend::createInstance<backend::QLinearMatMul>(std::string(name));
        //layer->init();    
        //layer->bind(_a_i, _a_scale_i, _a_zero_point_i, _b_i, _b_scale_i, _b_zero_point_i, _y_scale_i, _y_zero_point_i, _y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ConvInteger(py::module&);
#include "./layers/convinteger.h"
void init_layer_ConvInteger(py::module& m){
    m.def("_ConvInteger", [](py::str name, int _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _x_i , py::str _w_i , py::str _x_zero_point_i , py::str _w_zero_point_i , py::str _y_o) {
        auto layer = backend::createInstance<backend::ConvInteger>(std::string(name));
        //layer->init(_auto_pad, _dilations, _group, _kernel_shape, _pads, _strides);    
        //layer->bind(_x_i, _w_i, _x_zero_point_i, _w_zero_point_i, _y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_QLinearConv(py::module&);
#include "./layers/qlinearconv.h"
void init_layer_QLinearConv(py::module& m){
    m.def("_QLinearConv", [](py::str name, int _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _x_i , py::str _x_scale_i , py::str _x_zero_point_i , py::str _w_i , py::str _w_scale_i , py::str _w_zero_point_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _B_i , py::str _y_o) {
        auto layer = backend::createInstance<backend::QLinearConv>(std::string(name));
        //layer->init(_auto_pad, _dilations, _group, _kernel_shape, _pads, _strides);    
        //layer->bind(_x_i, _x_scale_i, _x_zero_point_i, _w_i, _w_scale_i, _w_zero_point_i, _y_scale_i, _y_zero_point_i, _B_i, _y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_QuantizeLinear(py::module&);
#include "./layers/quantizelinear.h"
void init_layer_QuantizeLinear(py::module& m){
    m.def("_QuantizeLinear", [](py::str name, py::str _x_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _y_o) {
        auto layer = backend::createInstance<backend::QuantizeLinear>(std::string(name));
        //layer->init();    
        //layer->bind(_x_i, _y_scale_i, _y_zero_point_i, _y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_DequantizeLinear(py::module&);
#include "./layers/dequantizelinear.h"
void init_layer_DequantizeLinear(py::module& m){
    m.def("_DequantizeLinear", [](py::str name, py::str _x_i , py::str _x_scale_i , py::str _x_zero_point_i , py::str _y_o) {
        auto layer = backend::createInstance<backend::DequantizeLinear>(std::string(name));
        //layer->init();    
        //layer->bind(_x_i, _x_scale_i, _x_zero_point_i, _y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_IsInf(py::module&);
#include "./layers/isinf.h"
void init_layer_IsInf(py::module& m){
    m.def("_IsInf", [](py::str name, int _detect_negative , int _detect_positive , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::IsInf>(std::string(name));
        //layer->init(_detect_negative, _detect_positive);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_RoiAlign(py::module&);
#include "./layers/roialign.h"
void init_layer_RoiAlign(py::module& m){
    m.def("_RoiAlign", [](py::str name, int _mode , int _output_height , int _output_width , int _sampling_ratio , float _spatial_scale , py::str _X_i , py::str _rois_i , py::str _batch_indices_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::RoiAlign>(std::string(name));
        //layer->init(_mode, _output_height, _output_width, _sampling_ratio, _spatial_scale);    
        //layer->bind(_X_i, _rois_i, _batch_indices_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ArrayFeatureExtractor(py::module&);
#include "./layers/arrayfeatureextractor.h"
void init_layer_ArrayFeatureExtractor(py::module& m){
    m.def("_ArrayFeatureExtractor", [](py::str name, py::str _X_i , py::str _Y_i , py::str _Z_o) {
        auto layer = backend::createInstance<backend::ArrayFeatureExtractor>(std::string(name));
        //layer->init();    
        //layer->bind(_X_i, _Y_i, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Binarizer(py::module&);
#include "./layers/binarizer.h"
void init_layer_Binarizer(py::module& m){
    m.def("_Binarizer", [](py::str name, float _threshold , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Binarizer>(std::string(name));
        //layer->init(_threshold);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_CategoryMapper(py::module&);
#include "./layers/categorymapper.h"
void init_layer_CategoryMapper(py::module& m){
    m.def("_CategoryMapper", [](py::str name, py::list _cats_int64s , int _default_int64 , int _default_string , py::str _cats_strings , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::CategoryMapper>(std::string(name));
        //layer->init(_cats_int64s, _default_int64, _default_string);    
        //layer->bind(_cats_strings, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_DictVectorizer(py::module&);
#include "./layers/dictvectorizer.h"
void init_layer_DictVectorizer(py::module& m){
    m.def("_DictVectorizer", [](py::str name, py::list _int64_vocabulary , py::str _string_vocabulary , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::DictVectorizer>(std::string(name));
        //layer->init(_int64_vocabulary);    
        //layer->bind(_string_vocabulary, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_FeatureVectorizer(py::module&);
#include "./layers/featurevectorizer.h"
void init_layer_FeatureVectorizer(py::module& m){
    m.def("_FeatureVectorizer", [](py::str name, py::list _inputdimensions , py::str _Y_o) {
        auto layer = backend::createInstance<backend::FeatureVectorizer>(std::string(name));
        //layer->init(_inputdimensions);    
        //layer->bind(_Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LabelEncoder(py::module&);
#include "./layers/labelencoder.h"
void init_layer_LabelEncoder(py::module& m){
    m.def("_LabelEncoder", [](py::str name, float _default_float , int _default_int64 , int _default_string , py::list _keys_int64s , py::list _values_int64s , py::str _keys_floats , py::str _keys_strings , py::str _values_floats , py::str _values_strings , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::LabelEncoder>(std::string(name));
        //layer->init(_default_float, _default_int64, _default_string, _keys_int64s, _values_int64s);    
        //layer->bind(_keys_floats, _keys_strings, _values_floats, _values_strings, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LinearClassifier(py::module&);
#include "./layers/linearclassifier.h"
void init_layer_LinearClassifier(py::module& m){
    m.def("_LinearClassifier", [](py::str name, py::list _classlabels_ints , int _multi_class , int _post_transform , py::str _coefficients , py::str _classlabels_strings , py::str _intercepts , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        auto layer = backend::createInstance<backend::LinearClassifier>(std::string(name));
        //layer->init(_classlabels_ints, _multi_class, _post_transform);    
        //layer->bind(_coefficients, _classlabels_strings, _intercepts, _X_i, _Y_o, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_LinearRegressor(py::module&);
#include "./layers/linearregressor.h"
void init_layer_LinearRegressor(py::module& m){
    m.def("_LinearRegressor", [](py::str name, int _post_transform , int _targets , py::str _coefficients , py::str _intercepts , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::LinearRegressor>(std::string(name));
        //layer->init(_post_transform, _targets);    
        //layer->bind(_coefficients, _intercepts, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Normalizer(py::module&);
#include "./layers/normalizer.h"
void init_layer_Normalizer(py::module& m){
    m.def("_Normalizer", [](py::str name, int _norm , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Normalizer>(std::string(name));
        //layer->init(_norm);    
        //layer->bind(_X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_SVMRegressor(py::module&);
#include "./layers/svmregressor.h"
void init_layer_SVMRegressor(py::module& m){
    m.def("_SVMRegressor", [](py::str name, int _kernel_type , int _n_supports , int _one_class , int _post_transform , py::str _coefficients , py::str _kernel_params , py::str _rho , py::str _support_vectors , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::SVMRegressor>(std::string(name));
        //layer->init(_kernel_type, _n_supports, _one_class, _post_transform);    
        //layer->bind(_coefficients, _kernel_params, _rho, _support_vectors, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_Scaler(py::module&);
#include "./layers/scaler.h"
void init_layer_Scaler(py::module& m){
    m.def("_Scaler", [](py::str name, py::str _offset , py::str _scale , py::str _X_i , py::str _Y_o) {
        auto layer = backend::createInstance<backend::Scaler>(std::string(name));
        //layer->init();    
        //layer->bind(_offset, _scale, _X_i, _Y_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_TreeEnsembleClassifier(py::module&);
#include "./layers/treeensembleclassifier.h"
void init_layer_TreeEnsembleClassifier(py::module& m){
    m.def("_TreeEnsembleClassifier", [](py::str name, py::list _class_ids , py::list _class_nodeids , py::list _class_treeids , py::list _classlabels_int64s , py::list _nodes_falsenodeids , py::list _nodes_featureids , py::list _nodes_missing_value_tracks_true , py::list _nodes_nodeids , py::list _nodes_treeids , py::list _nodes_truenodeids , int _post_transform , py::str _base_values , py::str _class_weights , py::str _classlabels_strings , py::str _nodes_hitrates , py::str _nodes_modes , py::str _nodes_values , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        auto layer = backend::createInstance<backend::TreeEnsembleClassifier>(std::string(name));
        //layer->init(_class_ids, _class_nodeids, _class_treeids, _classlabels_int64s, _nodes_falsenodeids, _nodes_featureids, _nodes_missing_value_tracks_true, _nodes_nodeids, _nodes_treeids, _nodes_truenodeids, _post_transform);    
        //layer->bind(_base_values, _class_weights, _classlabels_strings, _nodes_hitrates, _nodes_modes, _nodes_values, _X_i, _Y_o, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
void init_layer_ZipMap(py::module&);
#include "./layers/zipmap.h"
void init_layer_ZipMap(py::module& m){
    m.def("_ZipMap", [](py::str name, py::list _classlabels_int64s , py::str _classlabels_strings , py::str _X_i , py::str _Z_o) {
        auto layer = backend::createInstance<backend::ZipMap>(std::string(name));
        //layer->init(_classlabels_int64s);    
        //layer->bind(_classlabels_strings, _X_i, _Z_o); 
        backend::layer_dict[std::string(name)] = layer;
    });
}
