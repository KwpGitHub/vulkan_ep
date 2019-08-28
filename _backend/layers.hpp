void init_layer_LSTM(py::module&);
#include "./layers/lstm.h"
void init_layer_LSTM(py::module& m){
    m.def("_LSTM", [](py::str name, py::list _activation_alpha , py::list _activation_beta , py::list _activations , float _clip , py::str _direction , int _hidden_size , int _input_forget , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _initial_c_i , py::str _P_i , py::str _Y_o , py::str _Y_h_o , py::str _Y_c_o) {
        layers::LSTM* layer = new layers::LSTM(std::string(name));
        layer->init(backend::convert<float>(_activation_alpha), backend::convert<float>(_activation_beta), backend::convert<std::string>(_activations), _clip, _direction, _hidden_size, _input_forget);
        layer->bind(_X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _initial_c_i, _P_i, _Y_o, _Y_h_o, _Y_c_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LSTM" <<std::endl;

    });

    m.def("_LSTM_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LSTM" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Identity(py::module&);
#include "./layers/identity.h"
void init_layer_Identity(py::module& m){
    m.def("_Identity", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Identity* layer = new layers::Identity(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Identity" <<std::endl;

    });

    m.def("_Identity_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Identity" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Abs(py::module&);
#include "./layers/abs.h"
void init_layer_Abs(py::module& m){
    m.def("_Abs", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Abs* layer = new layers::Abs(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Abs" <<std::endl;

    });

    m.def("_Abs_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Abs" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_BatchNormalization(py::module&);
#include "./layers/batchnormalization.h"
void init_layer_BatchNormalization(py::module& m){
    m.def("_BatchNormalization", [](py::str name, float _epsilon , float _momentum , py::str _X_i , py::str _scale_i , py::str _B_i , py::str _mean_i , py::str _var_i , py::str _Y_o , py::str _mean_o , py::str _var_o , py::str _saved_mean_o , py::str _saved_var_o) {
        layers::BatchNormalization* layer = new layers::BatchNormalization(std::string(name));
        layer->init(_epsilon, _momentum);
        layer->bind(_X_i, _scale_i, _B_i, _mean_i, _var_i, _Y_o, _mean_o, _var_o, _saved_mean_o, _saved_var_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "BatchNormalization" <<std::endl;

    });

    m.def("_BatchNormalization_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: BatchNormalization" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Mean(py::module&);
#include "./layers/mean.h"
void init_layer_Mean(py::module& m){
    m.def("_Mean", [](py::str name, py::str _mean_o) {
        layers::Mean* layer = new layers::Mean(std::string(name));
        layer->init();
        layer->bind(_mean_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Mean" <<std::endl;

    });

    m.def("_Mean_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Mean" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Add(py::module&);
#include "./layers/add.h"
void init_layer_Add(py::module& m){
    m.def("_Add", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Add* layer = new layers::Add(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Add" <<std::endl;

    });

    m.def("_Add_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Add" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_GlobalMaxPool(py::module&);
#include "./layers/globalmaxpool.h"
void init_layer_GlobalMaxPool(py::module& m){
    m.def("_GlobalMaxPool", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::GlobalMaxPool* layer = new layers::GlobalMaxPool(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "GlobalMaxPool" <<std::endl;

    });

    m.def("_GlobalMaxPool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: GlobalMaxPool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Cast(py::module&);
#include "./layers/cast.h"
void init_layer_Cast(py::module& m){
    m.def("_Cast", [](py::str name, int _to , py::str _input_i , py::str _output_o) {
        layers::Cast* layer = new layers::Cast(std::string(name));
        layer->init(_to);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Cast" <<std::endl;

    });

    m.def("_Cast_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Cast" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_AveragePool(py::module&);
#include "./layers/averagepool.h"
void init_layer_AveragePool(py::module& m){
    m.def("_AveragePool", [](py::str name, py::list _kernel_shape , py::str _auto_pad , int _ceil_mode , int _count_include_pad , py::list _pads , py::list _strides , py::str _X_i , py::str _Y_o) {
        layers::AveragePool* layer = new layers::AveragePool(std::string(name));
        layer->init(backend::convert<int>(_kernel_shape), _auto_pad, _ceil_mode, _count_include_pad, backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "AveragePool" <<std::endl;

    });

    m.def("_AveragePool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: AveragePool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_And(py::module&);
#include "./layers/and.h"
void init_layer_And(py::module& m){
    m.def("_And", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::And* layer = new layers::And(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "And" <<std::endl;

    });

    m.def("_And_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: And" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LRN(py::module&);
#include "./layers/lrn.h"
void init_layer_LRN(py::module& m){
    m.def("_LRN", [](py::str name, int _size , float _alpha , float _beta , float _bias , py::str _X_i , py::str _Y_o) {
        layers::LRN* layer = new layers::LRN(std::string(name));
        layer->init(_size, _alpha, _beta, _bias);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LRN" <<std::endl;

    });

    m.def("_LRN_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LRN" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ArgMax(py::module&);
#include "./layers/argmax.h"
void init_layer_ArgMax(py::module& m){
    m.def("_ArgMax", [](py::str name, int _axis , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ArgMax* layer = new layers::ArgMax(std::string(name));
        layer->init(_axis, _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ArgMax" <<std::endl;

    });

    m.def("_ArgMax_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ArgMax" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Resize(py::module&);
#include "./layers/resize.h"
void init_layer_Resize(py::module& m){
    m.def("_Resize", [](py::str name, py::str _mode , py::str _X_i , py::str _scales_i , py::str _Y_o) {
        layers::Resize* layer = new layers::Resize(std::string(name));
        layer->init(_mode);
        layer->bind(_X_i, _scales_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Resize" <<std::endl;

    });

    m.def("_Resize_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Resize" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Expand(py::module&);
#include "./layers/expand.h"
void init_layer_Expand(py::module& m){
    m.def("_Expand", [](py::str name, py::str _input_i , py::str _shape_i , py::str _output_o) {
        layers::Expand* layer = new layers::Expand(std::string(name));
        layer->init();
        layer->bind(_input_i, _shape_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Expand" <<std::endl;

    });

    m.def("_Expand_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Expand" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Neg(py::module&);
#include "./layers/neg.h"
void init_layer_Neg(py::module& m){
    m.def("_Neg", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Neg* layer = new layers::Neg(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Neg" <<std::endl;

    });

    m.def("_Neg_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Neg" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Mul(py::module&);
#include "./layers/mul.h"
void init_layer_Mul(py::module& m){
    m.def("_Mul", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Mul* layer = new layers::Mul(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Mul" <<std::endl;

    });

    m.def("_Mul_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Mul" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ArgMin(py::module&);
#include "./layers/argmin.h"
void init_layer_ArgMin(py::module& m){
    m.def("_ArgMin", [](py::str name, int _axis , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ArgMin* layer = new layers::ArgMin(std::string(name));
        layer->init(_axis, _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ArgMin" <<std::endl;

    });

    m.def("_ArgMin_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ArgMin" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_CastMap(py::module&);
#include "./layers/castmap.h"
void init_layer_CastMap(py::module& m){
    m.def("_CastMap", [](py::str name, py::str _cast_to , py::str _map_form , int _max_map , py::str _X_i , py::str _Y_o) {
        layers::CastMap* layer = new layers::CastMap(std::string(name));
        layer->init(_cast_to, _map_form, _max_map);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "CastMap" <<std::endl;

    });

    m.def("_CastMap_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: CastMap" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Exp(py::module&);
#include "./layers/exp.h"
void init_layer_Exp(py::module& m){
    m.def("_Exp", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Exp* layer = new layers::Exp(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Exp" <<std::endl;

    });

    m.def("_Exp_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Exp" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Div(py::module&);
#include "./layers/div.h"
void init_layer_Div(py::module& m){
    m.def("_Div", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Div* layer = new layers::Div(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Div" <<std::endl;

    });

    m.def("_Div_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Div" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReverseSequence(py::module&);
#include "./layers/reversesequence.h"
void init_layer_ReverseSequence(py::module& m){
    m.def("_ReverseSequence", [](py::str name, int _batch_axis , int _time_axis , py::str _input_i , py::str _sequence_lens_i , py::str _Y_o) {
        layers::ReverseSequence* layer = new layers::ReverseSequence(std::string(name));
        layer->init(_batch_axis, _time_axis);
        layer->bind(_input_i, _sequence_lens_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReverseSequence" <<std::endl;

    });

    m.def("_ReverseSequence_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReverseSequence" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Ceil(py::module&);
#include "./layers/ceil.h"
void init_layer_Ceil(py::module& m){
    m.def("_Ceil", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Ceil* layer = new layers::Ceil(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Ceil" <<std::endl;

    });

    m.def("_Ceil_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Ceil" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_DepthToSpace(py::module&);
#include "./layers/depthtospace.h"
void init_layer_DepthToSpace(py::module& m){
    m.def("_DepthToSpace", [](py::str name, int _blocksize , py::str _input_i , py::str _output_o) {
        layers::DepthToSpace* layer = new layers::DepthToSpace(std::string(name));
        layer->init(_blocksize);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "DepthToSpace" <<std::endl;

    });

    m.def("_DepthToSpace_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: DepthToSpace" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Clip(py::module&);
#include "./layers/clip.h"
void init_layer_Clip(py::module& m){
    m.def("_Clip", [](py::str name, float _max , float _min , py::str _input_i , py::str _output_o) {
        layers::Clip* layer = new layers::Clip(std::string(name));
        layer->init(_max, _min);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Clip" <<std::endl;

    });

    m.def("_Clip_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Clip" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RNN(py::module&);
#include "./layers/rnn.h"
void init_layer_RNN(py::module& m){
    m.def("_RNN", [](py::str name, py::list _activation_alpha , py::list _activation_beta , py::list _activations , float _clip , py::str _direction , int _hidden_size , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _Y_o , py::str _Y_h_o) {
        layers::RNN* layer = new layers::RNN(std::string(name));
        layer->init(backend::convert<float>(_activation_alpha), backend::convert<float>(_activation_beta), backend::convert<std::string>(_activations), _clip, _direction, _hidden_size);
        layer->bind(_X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _Y_o, _Y_h_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RNN" <<std::endl;

    });

    m.def("_RNN_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RNN" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Concat(py::module&);
#include "./layers/concat.h"
void init_layer_Concat(py::module& m){
    m.def("_Concat", [](py::str name, int _axis , py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _concat_result_o) {
        layers::Concat* layer = new layers::Concat(std::string(name));
        layer->init(_axis);
        layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _concat_result_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Concat" <<std::endl;

    });

    m.def("_Concat_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Concat" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Constant(py::module&);
#include "./layers/constant.h"
void init_layer_Constant(py::module& m){
    m.def("_Constant", [](py::str name, py::list _value , py::str _output_o) {
        layers::Constant* layer = new layers::Constant(std::string(name));
        layer->init(backend::convert<float>(_value));
        layer->bind(_output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Constant" <<std::endl;

    });

    m.def("_Constant_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Constant" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LpPool(py::module&);
#include "./layers/lppool.h"
void init_layer_LpPool(py::module& m){
    m.def("_LpPool", [](py::str name, py::list _kernel_shape , py::str _auto_pad , int _p , py::list _pads , py::list _strides , py::str _X_i , py::str _Y_o) {
        layers::LpPool* layer = new layers::LpPool(std::string(name));
        layer->init(backend::convert<int>(_kernel_shape), _auto_pad, _p, backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LpPool" <<std::endl;

    });

    m.def("_LpPool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LpPool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Conv(py::module&);
#include "./layers/conv.h"
void init_layer_Conv(py::module& m){
    m.def("_Conv", [](py::str name, py::str _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _W_i , py::str _B_i , py::str _Y_o) {
        layers::Conv* layer = new layers::Conv(std::string(name));
        layer->init(_auto_pad, backend::convert<int>(_dilations), _group, backend::convert<int>(_kernel_shape), backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_X_i, _W_i, _B_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Conv" <<std::endl;

    });

    m.def("_Conv_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Conv" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Not(py::module&);
#include "./layers/not.h"
void init_layer_Not(py::module& m){
    m.def("_Not", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Not* layer = new layers::Not(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Not" <<std::endl;

    });

    m.def("_Not_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Not" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Gather(py::module&);
#include "./layers/gather.h"
void init_layer_Gather(py::module& m){
    m.def("_Gather", [](py::str name, int _axis , py::str _data_i , py::str _indices_i , py::str _output_o) {
        layers::Gather* layer = new layers::Gather(std::string(name));
        layer->init(_axis);
        layer->bind(_data_i, _indices_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Gather" <<std::endl;

    });

    m.def("_Gather_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Gather" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ConvTranspose(py::module&);
#include "./layers/convtranspose.h"
void init_layer_ConvTranspose(py::module& m){
    m.def("_ConvTranspose", [](py::str name, py::str _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _output_padding , py::list _output_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _W_i , py::str _B_i , py::str _Y_o) {
        layers::ConvTranspose* layer = new layers::ConvTranspose(std::string(name));
        layer->init(_auto_pad, backend::convert<int>(_dilations), _group, backend::convert<int>(_kernel_shape), backend::convert<int>(_output_padding), backend::convert<int>(_output_shape), backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_X_i, _W_i, _B_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ConvTranspose" <<std::endl;

    });

    m.def("_ConvTranspose_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ConvTranspose" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Dropout(py::module&);
#include "./layers/dropout.h"
void init_layer_Dropout(py::module& m){
    m.def("_Dropout", [](py::str name, float _ratio , py::str _data_i , py::str _output_o , py::str _mask_o) {
        layers::Dropout* layer = new layers::Dropout(std::string(name));
        layer->init(_ratio);
        layer->bind(_data_i, _output_o, _mask_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Dropout" <<std::endl;

    });

    m.def("_Dropout_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Dropout" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LeakyRelu(py::module&);
#include "./layers/leakyrelu.h"
void init_layer_LeakyRelu(py::module& m){
    m.def("_LeakyRelu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        layers::LeakyRelu* layer = new layers::LeakyRelu(std::string(name));
        layer->init(_alpha);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LeakyRelu" <<std::endl;

    });

    m.def("_LeakyRelu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LeakyRelu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Elu(py::module&);
#include "./layers/elu.h"
void init_layer_Elu(py::module& m){
    m.def("_Elu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        layers::Elu* layer = new layers::Elu(std::string(name));
        layer->init(_alpha);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Elu" <<std::endl;

    });

    m.def("_Elu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Elu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_GlobalAveragePool(py::module&);
#include "./layers/globalaveragepool.h"
void init_layer_GlobalAveragePool(py::module& m){
    m.def("_GlobalAveragePool", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::GlobalAveragePool* layer = new layers::GlobalAveragePool(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "GlobalAveragePool" <<std::endl;

    });

    m.def("_GlobalAveragePool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: GlobalAveragePool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Gemm(py::module&);
#include "./layers/gemm.h"
void init_layer_Gemm(py::module& m){
    m.def("_Gemm", [](py::str name, float _alpha , float _beta , int _transA , int _transB , py::str _A_i , py::str _B_i , py::str _C_i , py::str _Y_o) {
        layers::Gemm* layer = new layers::Gemm(std::string(name));
        layer->init(_alpha, _beta, _transA, _transB);
        layer->bind(_A_i, _B_i, _C_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Gemm" <<std::endl;

    });

    m.def("_Gemm_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Gemm" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MaxPool(py::module&);
#include "./layers/maxpool.h"
void init_layer_MaxPool(py::module& m){
    m.def("_MaxPool", [](py::str name, py::list _kernel_shape , py::str _auto_pad , int _ceil_mode , py::list _dilations , py::list _pads , int _storage_order , py::list _strides , py::str _X_i , py::str _Y_o , py::str _Indices_o) {
        layers::MaxPool* layer = new layers::MaxPool(std::string(name));
        layer->init(backend::convert<int>(_kernel_shape), _auto_pad, _ceil_mode, backend::convert<int>(_dilations), backend::convert<int>(_pads), _storage_order, backend::convert<int>(_strides));
        layer->bind(_X_i, _Y_o, _Indices_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MaxPool" <<std::endl;

    });

    m.def("_MaxPool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MaxPool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Equal(py::module&);
#include "./layers/equal.h"
void init_layer_Equal(py::module& m){
    m.def("_Equal", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Equal* layer = new layers::Equal(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Equal" <<std::endl;

    });

    m.def("_Equal_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Equal" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Tile(py::module&);
#include "./layers/tile.h"
void init_layer_Tile(py::module& m){
    m.def("_Tile", [](py::str name, py::str _input_i , py::str _repeats_i , py::str _output_o) {
        layers::Tile* layer = new layers::Tile(std::string(name));
        layer->init();
        layer->bind(_input_i, _repeats_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Tile" <<std::endl;

    });

    m.def("_Tile_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Tile" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Flatten(py::module&);
#include "./layers/flatten.h"
void init_layer_Flatten(py::module& m){
    m.def("_Flatten", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        layers::Flatten* layer = new layers::Flatten(std::string(name));
        layer->init(_axis);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Flatten" <<std::endl;

    });

    m.def("_Flatten_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Flatten" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Floor(py::module&);
#include "./layers/floor.h"
void init_layer_Floor(py::module& m){
    m.def("_Floor", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Floor* layer = new layers::Floor(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Floor" <<std::endl;

    });

    m.def("_Floor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Floor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_GRU(py::module&);
#include "./layers/gru.h"
void init_layer_GRU(py::module& m){
    m.def("_GRU", [](py::str name, py::list _activation_alpha , py::list _activation_beta , py::list _activations , float _clip , py::str _direction , int _hidden_size , int _linear_before_reset , py::str _X_i , py::str _W_i , py::str _R_i , py::str _B_i , py::str _sequence_lens_i , py::str _initial_h_i , py::str _Y_o , py::str _Y_h_o) {
        layers::GRU* layer = new layers::GRU(std::string(name));
        layer->init(backend::convert<float>(_activation_alpha), backend::convert<float>(_activation_beta), backend::convert<std::string>(_activations), _clip, _direction, _hidden_size, _linear_before_reset);
        layer->bind(_X_i, _W_i, _R_i, _B_i, _sequence_lens_i, _initial_h_i, _Y_o, _Y_h_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "GRU" <<std::endl;

    });

    m.def("_GRU_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: GRU" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_GlobalLpPool(py::module&);
#include "./layers/globallppool.h"
void init_layer_GlobalLpPool(py::module& m){
    m.def("_GlobalLpPool", [](py::str name, int _p , py::str _X_i , py::str _Y_o) {
        layers::GlobalLpPool* layer = new layers::GlobalLpPool(std::string(name));
        layer->init(_p);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "GlobalLpPool" <<std::endl;

    });

    m.def("_GlobalLpPool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: GlobalLpPool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Greater(py::module&);
#include "./layers/greater.h"
void init_layer_Greater(py::module& m){
    m.def("_Greater", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Greater* layer = new layers::Greater(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Greater" <<std::endl;

    });

    m.def("_Greater_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Greater" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_HardSigmoid(py::module&);
#include "./layers/hardsigmoid.h"
void init_layer_HardSigmoid(py::module& m){
    m.def("_HardSigmoid", [](py::str name, float _alpha , float _beta , py::str _X_i , py::str _Y_o) {
        layers::HardSigmoid* layer = new layers::HardSigmoid(std::string(name));
        layer->init(_alpha, _beta);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "HardSigmoid" <<std::endl;

    });

    m.def("_HardSigmoid_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: HardSigmoid" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Selu(py::module&);
#include "./layers/selu.h"
void init_layer_Selu(py::module& m){
    m.def("_Selu", [](py::str name, float _alpha , float _gamma , py::str _X_i , py::str _Y_o) {
        layers::Selu* layer = new layers::Selu(std::string(name));
        layer->init(_alpha, _gamma);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Selu" <<std::endl;

    });

    m.def("_Selu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Selu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Hardmax(py::module&);
#include "./layers/hardmax.h"
void init_layer_Hardmax(py::module& m){
    m.def("_Hardmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        layers::Hardmax* layer = new layers::Hardmax(std::string(name));
        layer->init(_axis);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Hardmax" <<std::endl;

    });

    m.def("_Hardmax_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Hardmax" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_If(py::module&);
#include "./layers/if.h"
void init_layer_If(py::module& m){
    m.def("_If", [](py::str name, int _else_branch , int _then_branch , py::str _cond_i) {
        layers::If* layer = new layers::If(std::string(name));
        layer->init(_else_branch, _then_branch);
        layer->bind(_cond_i);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "If" <<std::endl;

    });

    m.def("_If_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: If" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Min(py::module&);
#include "./layers/min.h"
void init_layer_Min(py::module& m){
    m.def("_Min", [](py::str name, py::str _min_o) {
        layers::Min* layer = new layers::Min(std::string(name));
        layer->init();
        layer->bind(_min_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Min" <<std::endl;

    });

    m.def("_Min_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Min" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_InstanceNormalization(py::module&);
#include "./layers/instancenormalization.h"
void init_layer_InstanceNormalization(py::module& m){
    m.def("_InstanceNormalization", [](py::str name, float _epsilon , py::str _input_i , py::str _scale_i , py::str _B_i , py::str _output_o) {
        layers::InstanceNormalization* layer = new layers::InstanceNormalization(std::string(name));
        layer->init(_epsilon);
        layer->bind(_input_i, _scale_i, _B_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "InstanceNormalization" <<std::endl;

    });

    m.def("_InstanceNormalization_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: InstanceNormalization" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Less(py::module&);
#include "./layers/less.h"
void init_layer_Less(py::module& m){
    m.def("_Less", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Less* layer = new layers::Less(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Less" <<std::endl;

    });

    m.def("_Less_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Less" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_EyeLike(py::module&);
#include "./layers/eyelike.h"
void init_layer_EyeLike(py::module& m){
    m.def("_EyeLike", [](py::str name, int _dtype , int _k , py::str _input_i , py::str _output_o) {
        layers::EyeLike* layer = new layers::EyeLike(std::string(name));
        layer->init(_dtype, _k);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "EyeLike" <<std::endl;

    });

    m.def("_EyeLike_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: EyeLike" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RandomNormal(py::module&);
#include "./layers/randomnormal.h"
void init_layer_RandomNormal(py::module& m){
    m.def("_RandomNormal", [](py::str name, py::list _shape , int _dtype , float _mean , float _scale , float _seed , py::str _output_o) {
        layers::RandomNormal* layer = new layers::RandomNormal(std::string(name));
        layer->init(backend::convert<int>(_shape), _dtype, _mean, _scale, _seed);
        layer->bind(_output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RandomNormal" <<std::endl;

    });

    m.def("_RandomNormal_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RandomNormal" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Slice(py::module&);
#include "./layers/slice.h"
void init_layer_Slice(py::module& m){
    m.def("_Slice", [](py::str name, py::str _data_i , py::str _starts_i , py::str _ends_i , py::str _axes_i , py::str _steps_i , py::str _output_o) {
        layers::Slice* layer = new layers::Slice(std::string(name));
        layer->init();
        layer->bind(_data_i, _starts_i, _ends_i, _axes_i, _steps_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Slice" <<std::endl;

    });

    m.def("_Slice_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Slice" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_PRelu(py::module&);
#include "./layers/prelu.h"
void init_layer_PRelu(py::module& m){
    m.def("_PRelu", [](py::str name, py::str _X_i , py::str _slope_i , py::str _Y_o) {
        layers::PRelu* layer = new layers::PRelu(std::string(name));
        layer->init();
        layer->bind(_X_i, _slope_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "PRelu" <<std::endl;

    });

    m.def("_PRelu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: PRelu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Log(py::module&);
#include "./layers/log.h"
void init_layer_Log(py::module& m){
    m.def("_Log", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Log* layer = new layers::Log(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Log" <<std::endl;

    });

    m.def("_Log_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Log" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LogSoftmax(py::module&);
#include "./layers/logsoftmax.h"
void init_layer_LogSoftmax(py::module& m){
    m.def("_LogSoftmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        layers::LogSoftmax* layer = new layers::LogSoftmax(std::string(name));
        layer->init(_axis);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LogSoftmax" <<std::endl;

    });

    m.def("_LogSoftmax_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LogSoftmax" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Loop(py::module&);
#include "./layers/loop.h"
void init_layer_Loop(py::module& m){
    m.def("_Loop", [](py::str name, int _body , py::str _M_i , py::str _cond_i) {
        layers::Loop* layer = new layers::Loop(std::string(name));
        layer->init(_body);
        layer->bind(_M_i, _cond_i);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Loop" <<std::endl;

    });

    m.def("_Loop_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Loop" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LpNormalization(py::module&);
#include "./layers/lpnormalization.h"
void init_layer_LpNormalization(py::module& m){
    m.def("_LpNormalization", [](py::str name, int _axis , int _p , py::str _input_i , py::str _output_o) {
        layers::LpNormalization* layer = new layers::LpNormalization(std::string(name));
        layer->init(_axis, _p);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LpNormalization" <<std::endl;

    });

    m.def("_LpNormalization_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LpNormalization" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MatMul(py::module&);
#include "./layers/matmul.h"
void init_layer_MatMul(py::module& m){
    m.def("_MatMul", [](py::str name, py::str _A_i , py::str _B_i , py::str _Y_o) {
        layers::MatMul* layer = new layers::MatMul(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MatMul" <<std::endl;

    });

    m.def("_MatMul_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MatMul" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceL2(py::module&);
#include "./layers/reducel2.h"
void init_layer_ReduceL2(py::module& m){
    m.def("_ReduceL2", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceL2* layer = new layers::ReduceL2(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceL2" <<std::endl;

    });

    m.def("_ReduceL2_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceL2" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Max(py::module&);
#include "./layers/max.h"
void init_layer_Max(py::module& m){
    m.def("_Max", [](py::str name, py::str _max_o) {
        layers::Max* layer = new layers::Max(std::string(name));
        layer->init();
        layer->bind(_max_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Max" <<std::endl;

    });

    m.def("_Max_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Max" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MaxRoiPool(py::module&);
#include "./layers/maxroipool.h"
void init_layer_MaxRoiPool(py::module& m){
    m.def("_MaxRoiPool", [](py::str name, py::list _pooled_shape , float _spatial_scale , py::str _X_i , py::str _rois_i , py::str _Y_o) {
        layers::MaxRoiPool* layer = new layers::MaxRoiPool(std::string(name));
        layer->init(backend::convert<int>(_pooled_shape), _spatial_scale);
        layer->bind(_X_i, _rois_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MaxRoiPool" <<std::endl;

    });

    m.def("_MaxRoiPool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MaxRoiPool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Or(py::module&);
#include "./layers/or.h"
void init_layer_Or(py::module& m){
    m.def("_Or", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Or* layer = new layers::Or(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Or" <<std::endl;

    });

    m.def("_Or_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Or" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Pad(py::module&);
#include "./layers/pad.h"
void init_layer_Pad(py::module& m){
    m.def("_Pad", [](py::str name, py::list _pads , py::str _mode , float _value , py::str _data_i , py::str _output_o) {
        layers::Pad* layer = new layers::Pad(std::string(name));
        layer->init(backend::convert<int>(_pads), _mode, _value);
        layer->bind(_data_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Pad" <<std::endl;

    });

    m.def("_Pad_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Pad" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RandomUniformLike(py::module&);
#include "./layers/randomuniformlike.h"
void init_layer_RandomUniformLike(py::module& m){
    m.def("_RandomUniformLike", [](py::str name, int _dtype , float _high , float _low , float _seed , py::str _input_i , py::str _output_o) {
        layers::RandomUniformLike* layer = new layers::RandomUniformLike(std::string(name));
        layer->init(_dtype, _high, _low, _seed);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RandomUniformLike" <<std::endl;

    });

    m.def("_RandomUniformLike_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RandomUniformLike" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Reciprocal(py::module&);
#include "./layers/reciprocal.h"
void init_layer_Reciprocal(py::module& m){
    m.def("_Reciprocal", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Reciprocal* layer = new layers::Reciprocal(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Reciprocal" <<std::endl;

    });

    m.def("_Reciprocal_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Reciprocal" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Pow(py::module&);
#include "./layers/pow.h"
void init_layer_Pow(py::module& m){
    m.def("_Pow", [](py::str name, py::str _X_i , py::str _Y_i , py::str _Z_o) {
        layers::Pow* layer = new layers::Pow(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_i, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Pow" <<std::endl;

    });

    m.def("_Pow_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Pow" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RandomNormalLike(py::module&);
#include "./layers/randomnormallike.h"
void init_layer_RandomNormalLike(py::module& m){
    m.def("_RandomNormalLike", [](py::str name, int _dtype , float _mean , float _scale , float _seed , py::str _input_i , py::str _output_o) {
        layers::RandomNormalLike* layer = new layers::RandomNormalLike(std::string(name));
        layer->init(_dtype, _mean, _scale, _seed);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RandomNormalLike" <<std::endl;

    });

    m.def("_RandomNormalLike_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RandomNormalLike" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_OneHot(py::module&);
#include "./layers/onehot.h"
void init_layer_OneHot(py::module& m){
    m.def("_OneHot", [](py::str name, int _axis , py::str _indices_i , py::str _depth_i , py::str _values_i , py::str _output_o) {
        layers::OneHot* layer = new layers::OneHot(std::string(name));
        layer->init(_axis);
        layer->bind(_indices_i, _depth_i, _values_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "OneHot" <<std::endl;

    });

    m.def("_OneHot_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: OneHot" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RandomUniform(py::module&);
#include "./layers/randomuniform.h"
void init_layer_RandomUniform(py::module& m){
    m.def("_RandomUniform", [](py::str name, py::list _shape , int _dtype , float _high , float _low , float _seed , py::str _output_o) {
        layers::RandomUniform* layer = new layers::RandomUniform(std::string(name));
        layer->init(backend::convert<int>(_shape), _dtype, _high, _low, _seed);
        layer->bind(_output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RandomUniform" <<std::endl;

    });

    m.def("_RandomUniform_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RandomUniform" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceL1(py::module&);
#include "./layers/reducel1.h"
void init_layer_ReduceL1(py::module& m){
    m.def("_ReduceL1", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceL1* layer = new layers::ReduceL1(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceL1" <<std::endl;

    });

    m.def("_ReduceL1_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceL1" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceLogSum(py::module&);
#include "./layers/reducelogsum.h"
void init_layer_ReduceLogSum(py::module& m){
    m.def("_ReduceLogSum", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceLogSum* layer = new layers::ReduceLogSum(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceLogSum" <<std::endl;

    });

    m.def("_ReduceLogSum_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceLogSum" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceLogSumExp(py::module&);
#include "./layers/reducelogsumexp.h"
void init_layer_ReduceLogSumExp(py::module& m){
    m.def("_ReduceLogSumExp", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceLogSumExp* layer = new layers::ReduceLogSumExp(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceLogSumExp" <<std::endl;

    });

    m.def("_ReduceLogSumExp_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceLogSumExp" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceMax(py::module&);
#include "./layers/reducemax.h"
void init_layer_ReduceMax(py::module& m){
    m.def("_ReduceMax", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceMax* layer = new layers::ReduceMax(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceMax" <<std::endl;

    });

    m.def("_ReduceMax_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceMax" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_OneHotEncoder(py::module&);
#include "./layers/onehotencoder.h"
void init_layer_OneHotEncoder(py::module& m){
    m.def("_OneHotEncoder", [](py::str name, py::list _cats_int64s , py::list _cats_strings , int _zeros , py::str _X_i , py::str _Y_o) {
        layers::OneHotEncoder* layer = new layers::OneHotEncoder(std::string(name));
        layer->init(backend::convert<int>(_cats_int64s), backend::convert<std::string>(_cats_strings), _zeros);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "OneHotEncoder" <<std::endl;

    });

    m.def("_OneHotEncoder_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: OneHotEncoder" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_IsNaN(py::module&);
#include "./layers/isnan.h"
void init_layer_IsNaN(py::module& m){
    m.def("_IsNaN", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::IsNaN* layer = new layers::IsNaN(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "IsNaN" <<std::endl;

    });

    m.def("_IsNaN_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: IsNaN" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceMean(py::module&);
#include "./layers/reducemean.h"
void init_layer_ReduceMean(py::module& m){
    m.def("_ReduceMean", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceMean* layer = new layers::ReduceMean(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceMean" <<std::endl;

    });

    m.def("_ReduceMean_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceMean" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceMin(py::module&);
#include "./layers/reducemin.h"
void init_layer_ReduceMin(py::module& m){
    m.def("_ReduceMin", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceMin* layer = new layers::ReduceMin(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceMin" <<std::endl;

    });

    m.def("_ReduceMin_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceMin" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_TreeEnsembleRegressor(py::module&);
#include "./layers/treeensembleregressor.h"
void init_layer_TreeEnsembleRegressor(py::module& m){
    m.def("_TreeEnsembleRegressor", [](py::str name, py::str _aggregate_function , py::list _base_values , int _n_targets , py::list _nodes_falsenodeids , py::list _nodes_featureids , py::list _nodes_hitrates , py::list _nodes_missing_value_tracks_true , py::list _nodes_modes , py::list _nodes_nodeids , py::list _nodes_treeids , py::list _nodes_truenodeids , py::list _nodes_values , py::str _post_transform , py::list _target_ids , py::list _target_nodeids , py::list _target_treeids , py::list _target_weights , py::str _X_i , py::str _Y_o) {
        layers::TreeEnsembleRegressor* layer = new layers::TreeEnsembleRegressor(std::string(name));
        layer->init(_aggregate_function, backend::convert<float>(_base_values), _n_targets, backend::convert<int>(_nodes_falsenodeids), backend::convert<int>(_nodes_featureids), backend::convert<float>(_nodes_hitrates), backend::convert<int>(_nodes_missing_value_tracks_true), backend::convert<std::string>(_nodes_modes), backend::convert<int>(_nodes_nodeids), backend::convert<int>(_nodes_treeids), backend::convert<int>(_nodes_truenodeids), backend::convert<float>(_nodes_values), _post_transform, backend::convert<int>(_target_ids), backend::convert<int>(_target_nodeids), backend::convert<int>(_target_treeids), backend::convert<float>(_target_weights));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "TreeEnsembleRegressor" <<std::endl;

    });

    m.def("_TreeEnsembleRegressor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: TreeEnsembleRegressor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceProd(py::module&);
#include "./layers/reduceprod.h"
void init_layer_ReduceProd(py::module& m){
    m.def("_ReduceProd", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceProd* layer = new layers::ReduceProd(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceProd" <<std::endl;

    });

    m.def("_ReduceProd_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceProd" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceSum(py::module&);
#include "./layers/reducesum.h"
void init_layer_ReduceSum(py::module& m){
    m.def("_ReduceSum", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceSum* layer = new layers::ReduceSum(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceSum" <<std::endl;

    });

    m.def("_ReduceSum_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceSum" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ReduceSumSquare(py::module&);
#include "./layers/reducesumsquare.h"
void init_layer_ReduceSumSquare(py::module& m){
    m.def("_ReduceSumSquare", [](py::str name, py::list _axes , int _keepdims , py::str _data_i , py::str _reduced_o) {
        layers::ReduceSumSquare* layer = new layers::ReduceSumSquare(std::string(name));
        layer->init(backend::convert<int>(_axes), _keepdims);
        layer->bind(_data_i, _reduced_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ReduceSumSquare" <<std::endl;

    });

    m.def("_ReduceSumSquare_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ReduceSumSquare" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Relu(py::module&);
#include "./layers/relu.h"
void init_layer_Relu(py::module& m){
    m.def("_Relu", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Relu* layer = new layers::Relu(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Relu" <<std::endl;

    });

    m.def("_Relu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Relu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Reshape(py::module&);
#include "./layers/reshape.h"
void init_layer_Reshape(py::module& m){
    m.def("_Reshape", [](py::str name, py::str _data_i , py::str _shape_i , py::str _reshaped_o) {
        layers::Reshape* layer = new layers::Reshape(std::string(name));
        layer->init();
        layer->bind(_data_i, _shape_i, _reshaped_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Reshape" <<std::endl;

    });

    m.def("_Reshape_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Reshape" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Shape(py::module&);
#include "./layers/shape.h"
void init_layer_Shape(py::module& m){
    m.def("_Shape", [](py::str name, py::str _data_i , py::str _shape_o) {
        layers::Shape* layer = new layers::Shape(std::string(name));
        layer->init();
        layer->bind(_data_i, _shape_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Shape" <<std::endl;

    });

    m.def("_Shape_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Shape" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sigmoid(py::module&);
#include "./layers/sigmoid.h"
void init_layer_Sigmoid(py::module& m){
    m.def("_Sigmoid", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Sigmoid* layer = new layers::Sigmoid(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sigmoid" <<std::endl;

    });

    m.def("_Sigmoid_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sigmoid" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Size(py::module&);
#include "./layers/size.h"
void init_layer_Size(py::module& m){
    m.def("_Size", [](py::str name, py::str _data_i , py::str _size_o) {
        layers::Size* layer = new layers::Size(std::string(name));
        layer->init();
        layer->bind(_data_i, _size_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Size" <<std::endl;

    });

    m.def("_Size_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Size" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Softmax(py::module&);
#include "./layers/softmax.h"
void init_layer_Softmax(py::module& m){
    m.def("_Softmax", [](py::str name, int _axis , py::str _input_i , py::str _output_o) {
        layers::Softmax* layer = new layers::Softmax(std::string(name));
        layer->init(_axis);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Softmax" <<std::endl;

    });

    m.def("_Softmax_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Softmax" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Softplus(py::module&);
#include "./layers/softplus.h"
void init_layer_Softplus(py::module& m){
    m.def("_Softplus", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Softplus* layer = new layers::Softplus(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Softplus" <<std::endl;

    });

    m.def("_Softplus_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Softplus" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Softsign(py::module&);
#include "./layers/softsign.h"
void init_layer_Softsign(py::module& m){
    m.def("_Softsign", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Softsign* layer = new layers::Softsign(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Softsign" <<std::endl;

    });

    m.def("_Softsign_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Softsign" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_SpaceToDepth(py::module&);
#include "./layers/spacetodepth.h"
void init_layer_SpaceToDepth(py::module& m){
    m.def("_SpaceToDepth", [](py::str name, int _blocksize , py::str _input_i , py::str _output_o) {
        layers::SpaceToDepth* layer = new layers::SpaceToDepth(std::string(name));
        layer->init(_blocksize);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "SpaceToDepth" <<std::endl;

    });

    m.def("_SpaceToDepth_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: SpaceToDepth" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_TfIdfVectorizer(py::module&);
#include "./layers/tfidfvectorizer.h"
void init_layer_TfIdfVectorizer(py::module& m){
    m.def("_TfIdfVectorizer", [](py::str name, int _max_gram_length , int _max_skip_count , int _min_gram_length , py::str _mode , py::list _ngram_counts , py::list _ngram_indexes , py::list _pool_int64s , py::list _pool_strings , py::list _weights , py::str _X_i , py::str _Y_o) {
        layers::TfIdfVectorizer* layer = new layers::TfIdfVectorizer(std::string(name));
        layer->init(_max_gram_length, _max_skip_count, _min_gram_length, _mode, backend::convert<int>(_ngram_counts), backend::convert<int>(_ngram_indexes), backend::convert<int>(_pool_int64s), backend::convert<std::string>(_pool_strings), backend::convert<float>(_weights));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "TfIdfVectorizer" <<std::endl;

    });

    m.def("_TfIdfVectorizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: TfIdfVectorizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Split(py::module&);
#include "./layers/split.h"
void init_layer_Split(py::module& m){
    m.def("_Split", [](py::str name, int _axis , py::list _split , py::str _input_i) {
        layers::Split* layer = new layers::Split(std::string(name));
        layer->init(_axis, backend::convert<int>(_split));
        layer->bind(_input_i);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Split" <<std::endl;

    });

    m.def("_Split_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Split" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Imputer(py::module&);
#include "./layers/imputer.h"
void init_layer_Imputer(py::module& m){
    m.def("_Imputer", [](py::str name, py::list _imputed_value_floats , py::list _imputed_value_int64s , float _replaced_value_float , int _replaced_value_int64 , py::str _X_i , py::str _Y_o) {
        layers::Imputer* layer = new layers::Imputer(std::string(name));
        layer->init(backend::convert<float>(_imputed_value_floats), backend::convert<int>(_imputed_value_int64s), _replaced_value_float, _replaced_value_int64);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Imputer" <<std::endl;

    });

    m.def("_Imputer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Imputer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sqrt(py::module&);
#include "./layers/sqrt.h"
void init_layer_Sqrt(py::module& m){
    m.def("_Sqrt", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::Sqrt* layer = new layers::Sqrt(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sqrt" <<std::endl;

    });

    m.def("_Sqrt_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sqrt" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Squeeze(py::module&);
#include "./layers/squeeze.h"
void init_layer_Squeeze(py::module& m){
    m.def("_Squeeze", [](py::str name, py::list _axes , py::str _data_i , py::str _squeezed_o) {
        layers::Squeeze* layer = new layers::Squeeze(std::string(name));
        layer->init(backend::convert<int>(_axes));
        layer->bind(_data_i, _squeezed_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Squeeze" <<std::endl;

    });

    m.def("_Squeeze_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Squeeze" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_TopK(py::module&);
#include "./layers/topk.h"
void init_layer_TopK(py::module& m){
    m.def("_TopK", [](py::str name, int _axis , py::str _X_i , py::str _K_i , py::str _Values_o , py::str _Indices_o) {
        layers::TopK* layer = new layers::TopK(std::string(name));
        layer->init(_axis);
        layer->bind(_X_i, _K_i, _Values_o, _Indices_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "TopK" <<std::endl;

    });

    m.def("_TopK_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: TopK" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sub(py::module&);
#include "./layers/sub.h"
void init_layer_Sub(py::module& m){
    m.def("_Sub", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Sub* layer = new layers::Sub(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sub" <<std::endl;

    });

    m.def("_Sub_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sub" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sum(py::module&);
#include "./layers/sum.h"
void init_layer_Sum(py::module& m){
    m.def("_Sum", [](py::str name, py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _sum_o) {
        layers::Sum* layer = new layers::Sum(std::string(name));
        layer->init();
        layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _sum_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sum" <<std::endl;

    });

    m.def("_Sum_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sum" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Shrink(py::module&);
#include "./layers/shrink.h"
void init_layer_Shrink(py::module& m){
    m.def("_Shrink", [](py::str name, float _bias , float _lambd , py::str _input_i , py::str _output_o) {
        layers::Shrink* layer = new layers::Shrink(std::string(name));
        layer->init(_bias, _lambd);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Shrink" <<std::endl;

    });

    m.def("_Shrink_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Shrink" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Tanh(py::module&);
#include "./layers/tanh.h"
void init_layer_Tanh(py::module& m){
    m.def("_Tanh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Tanh* layer = new layers::Tanh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Tanh" <<std::endl;

    });

    m.def("_Tanh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Tanh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Transpose(py::module&);
#include "./layers/transpose.h"
void init_layer_Transpose(py::module& m){
    m.def("_Transpose", [](py::str name, py::list _perm , py::str _data_i , py::str _transposed_o) {
        layers::Transpose* layer = new layers::Transpose(std::string(name));
        layer->init(backend::convert<int>(_perm));
        layer->bind(_data_i, _transposed_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Transpose" <<std::endl;

    });

    m.def("_Transpose_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Transpose" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Unsqueeze(py::module&);
#include "./layers/unsqueeze.h"
void init_layer_Unsqueeze(py::module& m){
    m.def("_Unsqueeze", [](py::str name, py::list _axes , py::str _data_i , py::str _expanded_o) {
        layers::Unsqueeze* layer = new layers::Unsqueeze(std::string(name));
        layer->init(backend::convert<int>(_axes));
        layer->bind(_data_i, _expanded_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Unsqueeze" <<std::endl;

    });

    m.def("_Unsqueeze_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Unsqueeze" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_SVMClassifier(py::module&);
#include "./layers/svmclassifier.h"
void init_layer_SVMClassifier(py::module& m){
    m.def("_SVMClassifier", [](py::str name, py::list _classlabels_ints , py::list _classlabels_strings , py::list _coefficients , py::list _kernel_params , py::str _kernel_type , py::str _post_transform , py::list _prob_a , py::list _prob_b , py::list _rho , py::list _support_vectors , py::list _vectors_per_class , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        layers::SVMClassifier* layer = new layers::SVMClassifier(std::string(name));
        layer->init(backend::convert<int>(_classlabels_ints), backend::convert<std::string>(_classlabels_strings), backend::convert<float>(_coefficients), backend::convert<float>(_kernel_params), _kernel_type, _post_transform, backend::convert<float>(_prob_a), backend::convert<float>(_prob_b), backend::convert<float>(_rho), backend::convert<float>(_support_vectors), backend::convert<int>(_vectors_per_class));
        layer->bind(_X_i, _Y_o, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "SVMClassifier" <<std::endl;

    });

    m.def("_SVMClassifier_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: SVMClassifier" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Xor(py::module&);
#include "./layers/xor.h"
void init_layer_Xor(py::module& m){
    m.def("_Xor", [](py::str name, py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Xor* layer = new layers::Xor(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Xor" <<std::endl;

    });

    m.def("_Xor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Xor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Acos(py::module&);
#include "./layers/acos.h"
void init_layer_Acos(py::module& m){
    m.def("_Acos", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Acos* layer = new layers::Acos(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Acos" <<std::endl;

    });

    m.def("_Acos_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Acos" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Asin(py::module&);
#include "./layers/asin.h"
void init_layer_Asin(py::module& m){
    m.def("_Asin", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Asin* layer = new layers::Asin(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Asin" <<std::endl;

    });

    m.def("_Asin_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Asin" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Atan(py::module&);
#include "./layers/atan.h"
void init_layer_Atan(py::module& m){
    m.def("_Atan", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Atan* layer = new layers::Atan(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Atan" <<std::endl;

    });

    m.def("_Atan_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Atan" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Cos(py::module&);
#include "./layers/cos.h"
void init_layer_Cos(py::module& m){
    m.def("_Cos", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Cos* layer = new layers::Cos(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Cos" <<std::endl;

    });

    m.def("_Cos_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Cos" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sin(py::module&);
#include "./layers/sin.h"
void init_layer_Sin(py::module& m){
    m.def("_Sin", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Sin* layer = new layers::Sin(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sin" <<std::endl;

    });

    m.def("_Sin_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sin" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Tan(py::module&);
#include "./layers/tan.h"
void init_layer_Tan(py::module& m){
    m.def("_Tan", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Tan* layer = new layers::Tan(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Tan" <<std::endl;

    });

    m.def("_Tan_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Tan" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Multinomial(py::module&);
#include "./layers/multinomial.h"
void init_layer_Multinomial(py::module& m){
    m.def("_Multinomial", [](py::str name, int _dtype , int _sample_size , float _seed , py::str _input_i , py::str _output_o) {
        layers::Multinomial* layer = new layers::Multinomial(std::string(name));
        layer->init(_dtype, _sample_size, _seed);
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Multinomial" <<std::endl;

    });

    m.def("_Multinomial_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Multinomial" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Scan(py::module&);
#include "./layers/scan.h"
void init_layer_Scan(py::module& m){
    m.def("_Scan", [](py::str name, int _body , int _num_scan_inputs , py::list _scan_input_axes , py::list _scan_input_directions , py::list _scan_output_axes , py::list _scan_output_directions , py::str _x0_i , py::str _x1_i , py::str _x2_i , py::str _x3_i , py::str _x4_i , py::str _x5_i , py::str _x6_i , py::str _x7_i , py::str _x8_i , py::str _x9_i , py::str _x10_i , py::str _x11_i , py::str _x12_i , py::str _x13_i , py::str _x14_i , py::str _x15_i , py::str _x16_i , py::str _x17_i , py::str _x18_i , py::str _x19_i , py::str _x20_i , py::str _x21_i , py::str _x22_i , py::str _x23_i , py::str _x24_i , py::str _x25_i , py::str _x26_i , py::str _x27_i , py::str _x28_i , py::str _x29_i , py::str _x30_i , py::str _x31_i , py::str _y0_o , py::str _y1_o , py::str _y2_o , py::str _y3_o , py::str _y4_o , py::str _y5_o , py::str _y6_o , py::str _y7_o , py::str _y8_o , py::str _y9_o , py::str _y10_o , py::str _y11_o , py::str _y12_o , py::str _y13_o , py::str _y14_o , py::str _y15_o , py::str _y16_o , py::str _y17_o , py::str _y18_o , py::str _y19_o , py::str _y20_o , py::str _y21_o , py::str _y22_o , py::str _y23_o , py::str _y24_o , py::str _y25_o , py::str _y26_o , py::str _y27_o , py::str _y28_o , py::str _y29_o , py::str _y30_o , py::str _y31_o) {
        layers::Scan* layer = new layers::Scan(std::string(name));
        layer->init(_body, _num_scan_inputs, backend::convert<int>(_scan_input_axes), backend::convert<int>(_scan_input_directions), backend::convert<int>(_scan_output_axes), backend::convert<int>(_scan_output_directions));
        layer->bind(_x0_i, _x1_i, _x2_i, _x3_i, _x4_i, _x5_i, _x6_i, _x7_i, _x8_i, _x9_i, _x10_i, _x11_i, _x12_i, _x13_i, _x14_i, _x15_i, _x16_i, _x17_i, _x18_i, _x19_i, _x20_i, _x21_i, _x22_i, _x23_i, _x24_i, _x25_i, _x26_i, _x27_i, _x28_i, _x29_i, _x30_i, _x31_i, _y0_o, _y1_o, _y2_o, _y3_o, _y4_o, _y5_o, _y6_o, _y7_o, _y8_o, _y9_o, _y10_o, _y11_o, _y12_o, _y13_o, _y14_o, _y15_o, _y16_o, _y17_o, _y18_o, _y19_o, _y20_o, _y21_o, _y22_o, _y23_o, _y24_o, _y25_o, _y26_o, _y27_o, _y28_o, _y29_o, _y30_o, _y31_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Scan" <<std::endl;

    });

    m.def("_Scan_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Scan" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Compress(py::module&);
#include "./layers/compress.h"
void init_layer_Compress(py::module& m){
    m.def("_Compress", [](py::str name, int _axis , py::str _input_i , py::str _condition_i , py::str _output_o) {
        layers::Compress* layer = new layers::Compress(std::string(name));
        layer->init(_axis);
        layer->bind(_input_i, _condition_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Compress" <<std::endl;

    });

    m.def("_Compress_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Compress" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ConstantOfShape(py::module&);
#include "./layers/constantofshape.h"
void init_layer_ConstantOfShape(py::module& m){
    m.def("_ConstantOfShape", [](py::str name, py::list _value , py::str _input_i , py::str _output_o) {
        layers::ConstantOfShape* layer = new layers::ConstantOfShape(std::string(name));
        layer->init(backend::convert<float>(_value));
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ConstantOfShape" <<std::endl;

    });

    m.def("_ConstantOfShape_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ConstantOfShape" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MaxUnpool(py::module&);
#include "./layers/maxunpool.h"
void init_layer_MaxUnpool(py::module& m){
    m.def("_MaxUnpool", [](py::str name, py::list _kernel_shape , py::list _pads , py::list _strides , py::str _X_i , py::str _I_i , py::str _output_shape_i , py::str _output_o) {
        layers::MaxUnpool* layer = new layers::MaxUnpool(std::string(name));
        layer->init(backend::convert<int>(_kernel_shape), backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_X_i, _I_i, _output_shape_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MaxUnpool" <<std::endl;

    });

    m.def("_MaxUnpool_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MaxUnpool" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Scatter(py::module&);
#include "./layers/scatter.h"
void init_layer_Scatter(py::module& m){
    m.def("_Scatter", [](py::str name, int _axis , py::str _data_i , py::str _indices_i , py::str _updates_i , py::str _output_o) {
        layers::Scatter* layer = new layers::Scatter(std::string(name));
        layer->init(_axis);
        layer->bind(_data_i, _indices_i, _updates_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Scatter" <<std::endl;

    });

    m.def("_Scatter_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Scatter" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sinh(py::module&);
#include "./layers/sinh.h"
void init_layer_Sinh(py::module& m){
    m.def("_Sinh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Sinh* layer = new layers::Sinh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sinh" <<std::endl;

    });

    m.def("_Sinh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sinh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Cosh(py::module&);
#include "./layers/cosh.h"
void init_layer_Cosh(py::module& m){
    m.def("_Cosh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Cosh* layer = new layers::Cosh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Cosh" <<std::endl;

    });

    m.def("_Cosh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Cosh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Asinh(py::module&);
#include "./layers/asinh.h"
void init_layer_Asinh(py::module& m){
    m.def("_Asinh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Asinh* layer = new layers::Asinh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Asinh" <<std::endl;

    });

    m.def("_Asinh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Asinh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Acosh(py::module&);
#include "./layers/acosh.h"
void init_layer_Acosh(py::module& m){
    m.def("_Acosh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Acosh* layer = new layers::Acosh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Acosh" <<std::endl;

    });

    m.def("_Acosh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Acosh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_NonMaxSuppression(py::module&);
#include "./layers/nonmaxsuppression.h"
void init_layer_NonMaxSuppression(py::module& m){
    m.def("_NonMaxSuppression", [](py::str name, int _center_point_box , py::str _boxes_i , py::str _scores_i , py::str _max_output_boxes_per_class_i , py::str _iou_threshold_i , py::str _score_threshold_i , py::str _selected_indices_o) {
        layers::NonMaxSuppression* layer = new layers::NonMaxSuppression(std::string(name));
        layer->init(_center_point_box);
        layer->bind(_boxes_i, _scores_i, _max_output_boxes_per_class_i, _iou_threshold_i, _score_threshold_i, _selected_indices_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "NonMaxSuppression" <<std::endl;

    });

    m.def("_NonMaxSuppression_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: NonMaxSuppression" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Atanh(py::module&);
#include "./layers/atanh.h"
void init_layer_Atanh(py::module& m){
    m.def("_Atanh", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Atanh* layer = new layers::Atanh(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Atanh" <<std::endl;

    });

    m.def("_Atanh_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Atanh" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Sign(py::module&);
#include "./layers/sign.h"
void init_layer_Sign(py::module& m){
    m.def("_Sign", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Sign* layer = new layers::Sign(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Sign" <<std::endl;

    });

    m.def("_Sign_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Sign" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Erf(py::module&);
#include "./layers/erf.h"
void init_layer_Erf(py::module& m){
    m.def("_Erf", [](py::str name, py::str _input_i , py::str _output_o) {
        layers::Erf* layer = new layers::Erf(std::string(name));
        layer->init();
        layer->bind(_input_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Erf" <<std::endl;

    });

    m.def("_Erf_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Erf" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Where(py::module&);
#include "./layers/where.h"
void init_layer_Where(py::module& m){
    m.def("_Where", [](py::str name, py::str _condition_i , py::str _X_i , py::str _Y_i , py::str _output_o) {
        layers::Where* layer = new layers::Where(std::string(name));
        layer->init();
        layer->bind(_condition_i, _X_i, _Y_i, _output_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Where" <<std::endl;

    });

    m.def("_Where_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Where" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_NonZero(py::module&);
#include "./layers/nonzero.h"
void init_layer_NonZero(py::module& m){
    m.def("_NonZero", [](py::str name, py::str _X_i , py::str _Y_o) {
        layers::NonZero* layer = new layers::NonZero(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "NonZero" <<std::endl;

    });

    m.def("_NonZero_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: NonZero" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MeanVarianceNormalization(py::module&);
#include "./layers/meanvariancenormalization.h"
void init_layer_MeanVarianceNormalization(py::module& m){
    m.def("_MeanVarianceNormalization", [](py::str name, py::list _axes , py::str _X_i , py::str _Y_o) {
        layers::MeanVarianceNormalization* layer = new layers::MeanVarianceNormalization(std::string(name));
        layer->init(backend::convert<int>(_axes));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MeanVarianceNormalization" <<std::endl;

    });

    m.def("_MeanVarianceNormalization_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MeanVarianceNormalization" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_StringNormalizer(py::module&);
#include "./layers/stringnormalizer.h"
void init_layer_StringNormalizer(py::module& m){
    m.def("_StringNormalizer", [](py::str name, py::str _case_change_action , int _is_case_sensitive , py::str _locale , py::list _stopwords , py::str _X_i , py::str _Y_o) {
        layers::StringNormalizer* layer = new layers::StringNormalizer(std::string(name));
        layer->init(_case_change_action, _is_case_sensitive, _locale, backend::convert<std::string>(_stopwords));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "StringNormalizer" <<std::endl;

    });

    m.def("_StringNormalizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: StringNormalizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Mod(py::module&);
#include "./layers/mod.h"
void init_layer_Mod(py::module& m){
    m.def("_Mod", [](py::str name, int _fmod , py::str _A_i , py::str _B_i , py::str _C_o) {
        layers::Mod* layer = new layers::Mod(std::string(name));
        layer->init(_fmod);
        layer->bind(_A_i, _B_i, _C_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Mod" <<std::endl;

    });

    m.def("_Mod_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Mod" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ThresholdedRelu(py::module&);
#include "./layers/thresholdedrelu.h"
void init_layer_ThresholdedRelu(py::module& m){
    m.def("_ThresholdedRelu", [](py::str name, float _alpha , py::str _X_i , py::str _Y_o) {
        layers::ThresholdedRelu* layer = new layers::ThresholdedRelu(std::string(name));
        layer->init(_alpha);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ThresholdedRelu" <<std::endl;

    });

    m.def("_ThresholdedRelu_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ThresholdedRelu" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_MatMulInteger(py::module&);
#include "./layers/matmulinteger.h"
void init_layer_MatMulInteger(py::module& m){
    m.def("_MatMulInteger", [](py::str name, py::str _A_i , py::str _B_i , py::str _a_zero_point_i , py::str _b_zero_point_i , py::str _Y_o) {
        layers::MatMulInteger* layer = new layers::MatMulInteger(std::string(name));
        layer->init();
        layer->bind(_A_i, _B_i, _a_zero_point_i, _b_zero_point_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "MatMulInteger" <<std::endl;

    });

    m.def("_MatMulInteger_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: MatMulInteger" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_QLinearMatMul(py::module&);
#include "./layers/qlinearmatmul.h"
void init_layer_QLinearMatMul(py::module& m){
    m.def("_QLinearMatMul", [](py::str name, py::str _a_i , py::str _a_scale_i , py::str _a_zero_point_i , py::str _b_i , py::str _b_scale_i , py::str _b_zero_point_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _y_o) {
        layers::QLinearMatMul* layer = new layers::QLinearMatMul(std::string(name));
        layer->init();
        layer->bind(_a_i, _a_scale_i, _a_zero_point_i, _b_i, _b_scale_i, _b_zero_point_i, _y_scale_i, _y_zero_point_i, _y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "QLinearMatMul" <<std::endl;

    });

    m.def("_QLinearMatMul_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: QLinearMatMul" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ConvInteger(py::module&);
#include "./layers/convinteger.h"
void init_layer_ConvInteger(py::module& m){
    m.def("_ConvInteger", [](py::str name, py::str _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _x_i , py::str _w_i , py::str _x_zero_point_i , py::str _w_zero_point_i , py::str _y_o) {
        layers::ConvInteger* layer = new layers::ConvInteger(std::string(name));
        layer->init(_auto_pad, backend::convert<int>(_dilations), _group, backend::convert<int>(_kernel_shape), backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_x_i, _w_i, _x_zero_point_i, _w_zero_point_i, _y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ConvInteger" <<std::endl;

    });

    m.def("_ConvInteger_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ConvInteger" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_QLinearConv(py::module&);
#include "./layers/qlinearconv.h"
void init_layer_QLinearConv(py::module& m){
    m.def("_QLinearConv", [](py::str name, py::str _auto_pad , py::list _dilations , int _group , py::list _kernel_shape , py::list _pads , py::list _strides , py::str _x_i , py::str _x_scale_i , py::str _x_zero_point_i , py::str _w_i , py::str _w_scale_i , py::str _w_zero_point_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _B_i , py::str _y_o) {
        layers::QLinearConv* layer = new layers::QLinearConv(std::string(name));
        layer->init(_auto_pad, backend::convert<int>(_dilations), _group, backend::convert<int>(_kernel_shape), backend::convert<int>(_pads), backend::convert<int>(_strides));
        layer->bind(_x_i, _x_scale_i, _x_zero_point_i, _w_i, _w_scale_i, _w_zero_point_i, _y_scale_i, _y_zero_point_i, _B_i, _y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "QLinearConv" <<std::endl;

    });

    m.def("_QLinearConv_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: QLinearConv" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_QuantizeLinear(py::module&);
#include "./layers/quantizelinear.h"
void init_layer_QuantizeLinear(py::module& m){
    m.def("_QuantizeLinear", [](py::str name, py::str _x_i , py::str _y_scale_i , py::str _y_zero_point_i , py::str _y_o) {
        layers::QuantizeLinear* layer = new layers::QuantizeLinear(std::string(name));
        layer->init();
        layer->bind(_x_i, _y_scale_i, _y_zero_point_i, _y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "QuantizeLinear" <<std::endl;

    });

    m.def("_QuantizeLinear_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: QuantizeLinear" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_DequantizeLinear(py::module&);
#include "./layers/dequantizelinear.h"
void init_layer_DequantizeLinear(py::module& m){
    m.def("_DequantizeLinear", [](py::str name, py::str _x_i , py::str _x_scale_i , py::str _x_zero_point_i , py::str _y_o) {
        layers::DequantizeLinear* layer = new layers::DequantizeLinear(std::string(name));
        layer->init();
        layer->bind(_x_i, _x_scale_i, _x_zero_point_i, _y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "DequantizeLinear" <<std::endl;

    });

    m.def("_DequantizeLinear_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: DequantizeLinear" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_IsInf(py::module&);
#include "./layers/isinf.h"
void init_layer_IsInf(py::module& m){
    m.def("_IsInf", [](py::str name, int _detect_negative , int _detect_positive , py::str _X_i , py::str _Y_o) {
        layers::IsInf* layer = new layers::IsInf(std::string(name));
        layer->init(_detect_negative, _detect_positive);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "IsInf" <<std::endl;

    });

    m.def("_IsInf_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: IsInf" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_RoiAlign(py::module&);
#include "./layers/roialign.h"
void init_layer_RoiAlign(py::module& m){
    m.def("_RoiAlign", [](py::str name, py::str _mode , int _output_height , int _output_width , int _sampling_ratio , float _spatial_scale , py::str _X_i , py::str _rois_i , py::str _batch_indices_i , py::str _Y_o) {
        layers::RoiAlign* layer = new layers::RoiAlign(std::string(name));
        layer->init(_mode, _output_height, _output_width, _sampling_ratio, _spatial_scale);
        layer->bind(_X_i, _rois_i, _batch_indices_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "RoiAlign" <<std::endl;

    });

    m.def("_RoiAlign_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: RoiAlign" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ArrayFeatureExtractor(py::module&);
#include "./layers/arrayfeatureextractor.h"
void init_layer_ArrayFeatureExtractor(py::module& m){
    m.def("_ArrayFeatureExtractor", [](py::str name, py::str _X_i , py::str _Y_i , py::str _Z_o) {
        layers::ArrayFeatureExtractor* layer = new layers::ArrayFeatureExtractor(std::string(name));
        layer->init();
        layer->bind(_X_i, _Y_i, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ArrayFeatureExtractor" <<std::endl;

    });

    m.def("_ArrayFeatureExtractor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ArrayFeatureExtractor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Binarizer(py::module&);
#include "./layers/binarizer.h"
void init_layer_Binarizer(py::module& m){
    m.def("_Binarizer", [](py::str name, float _threshold , py::str _X_i , py::str _Y_o) {
        layers::Binarizer* layer = new layers::Binarizer(std::string(name));
        layer->init(_threshold);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Binarizer" <<std::endl;

    });

    m.def("_Binarizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Binarizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_CategoryMapper(py::module&);
#include "./layers/categorymapper.h"
void init_layer_CategoryMapper(py::module& m){
    m.def("_CategoryMapper", [](py::str name, py::list _cats_int64s , py::list _cats_strings , int _default_int64 , py::str _default_string , py::str _X_i , py::str _Y_o) {
        layers::CategoryMapper* layer = new layers::CategoryMapper(std::string(name));
        layer->init(backend::convert<int>(_cats_int64s), backend::convert<std::string>(_cats_strings), _default_int64, _default_string);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "CategoryMapper" <<std::endl;

    });

    m.def("_CategoryMapper_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: CategoryMapper" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_DictVectorizer(py::module&);
#include "./layers/dictvectorizer.h"
void init_layer_DictVectorizer(py::module& m){
    m.def("_DictVectorizer", [](py::str name, py::list _int64_vocabulary , py::list _string_vocabulary , py::str _X_i , py::str _Y_o) {
        layers::DictVectorizer* layer = new layers::DictVectorizer(std::string(name));
        layer->init(backend::convert<int>(_int64_vocabulary), backend::convert<std::string>(_string_vocabulary));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "DictVectorizer" <<std::endl;

    });

    m.def("_DictVectorizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: DictVectorizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_FeatureVectorizer(py::module&);
#include "./layers/featurevectorizer.h"
void init_layer_FeatureVectorizer(py::module& m){
    m.def("_FeatureVectorizer", [](py::str name, py::list _inputdimensions , py::str _Y_o) {
        layers::FeatureVectorizer* layer = new layers::FeatureVectorizer(std::string(name));
        layer->init(backend::convert<int>(_inputdimensions));
        layer->bind(_Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "FeatureVectorizer" <<std::endl;

    });

    m.def("_FeatureVectorizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: FeatureVectorizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LabelEncoder(py::module&);
#include "./layers/labelencoder.h"
void init_layer_LabelEncoder(py::module& m){
    m.def("_LabelEncoder", [](py::str name, float _default_float , int _default_int64 , py::str _default_string , py::list _keys_floats , py::list _keys_int64s , py::list _keys_strings , py::list _values_floats , py::list _values_int64s , py::list _values_strings , py::str _X_i , py::str _Y_o) {
        layers::LabelEncoder* layer = new layers::LabelEncoder(std::string(name));
        layer->init(_default_float, _default_int64, _default_string, backend::convert<float>(_keys_floats), backend::convert<int>(_keys_int64s), backend::convert<std::string>(_keys_strings), backend::convert<float>(_values_floats), backend::convert<int>(_values_int64s), backend::convert<std::string>(_values_strings));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LabelEncoder" <<std::endl;

    });

    m.def("_LabelEncoder_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LabelEncoder" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LinearClassifier(py::module&);
#include "./layers/linearclassifier.h"
void init_layer_LinearClassifier(py::module& m){
    m.def("_LinearClassifier", [](py::str name, py::list _coefficients , py::list _classlabels_ints , py::list _classlabels_strings , py::list _intercepts , int _multi_class , py::str _post_transform , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        layers::LinearClassifier* layer = new layers::LinearClassifier(std::string(name));
        layer->init(backend::convert<float>(_coefficients), backend::convert<int>(_classlabels_ints), backend::convert<std::string>(_classlabels_strings), backend::convert<float>(_intercepts), _multi_class, _post_transform);
        layer->bind(_X_i, _Y_o, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LinearClassifier" <<std::endl;

    });

    m.def("_LinearClassifier_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LinearClassifier" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_LinearRegressor(py::module&);
#include "./layers/linearregressor.h"
void init_layer_LinearRegressor(py::module& m){
    m.def("_LinearRegressor", [](py::str name, py::list _coefficients , py::list _intercepts , py::str _post_transform , int _targets , py::str _X_i , py::str _Y_o) {
        layers::LinearRegressor* layer = new layers::LinearRegressor(std::string(name));
        layer->init(backend::convert<float>(_coefficients), backend::convert<float>(_intercepts), _post_transform, _targets);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "LinearRegressor" <<std::endl;

    });

    m.def("_LinearRegressor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: LinearRegressor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Normalizer(py::module&);
#include "./layers/normalizer.h"
void init_layer_Normalizer(py::module& m){
    m.def("_Normalizer", [](py::str name, py::str _norm , py::str _X_i , py::str _Y_o) {
        layers::Normalizer* layer = new layers::Normalizer(std::string(name));
        layer->init(_norm);
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Normalizer" <<std::endl;

    });

    m.def("_Normalizer_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Normalizer" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_SVMRegressor(py::module&);
#include "./layers/svmregressor.h"
void init_layer_SVMRegressor(py::module& m){
    m.def("_SVMRegressor", [](py::str name, py::list _coefficients , py::list _kernel_params , py::str _kernel_type , int _n_supports , int _one_class , py::str _post_transform , py::list _rho , py::list _support_vectors , py::str _X_i , py::str _Y_o) {
        layers::SVMRegressor* layer = new layers::SVMRegressor(std::string(name));
        layer->init(backend::convert<float>(_coefficients), backend::convert<float>(_kernel_params), _kernel_type, _n_supports, _one_class, _post_transform, backend::convert<float>(_rho), backend::convert<float>(_support_vectors));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "SVMRegressor" <<std::endl;

    });

    m.def("_SVMRegressor_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: SVMRegressor" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_Scaler(py::module&);
#include "./layers/scaler.h"
void init_layer_Scaler(py::module& m){
    m.def("_Scaler", [](py::str name, py::list _offset , py::list _scale , py::str _X_i , py::str _Y_o) {
        layers::Scaler* layer = new layers::Scaler(std::string(name));
        layer->init(backend::convert<float>(_offset), backend::convert<float>(_scale));
        layer->bind(_X_i, _Y_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "Scaler" <<std::endl;

    });

    m.def("_Scaler_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: Scaler" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_TreeEnsembleClassifier(py::module&);
#include "./layers/treeensembleclassifier.h"
void init_layer_TreeEnsembleClassifier(py::module& m){
    m.def("_TreeEnsembleClassifier", [](py::str name, py::list _base_values , py::list _class_ids , py::list _class_nodeids , py::list _class_treeids , py::list _class_weights , py::list _classlabels_int64s , py::list _classlabels_strings , py::list _nodes_falsenodeids , py::list _nodes_featureids , py::list _nodes_hitrates , py::list _nodes_missing_value_tracks_true , py::list _nodes_modes , py::list _nodes_nodeids , py::list _nodes_treeids , py::list _nodes_truenodeids , py::list _nodes_values , py::str _post_transform , py::str _X_i , py::str _Y_o , py::str _Z_o) {
        layers::TreeEnsembleClassifier* layer = new layers::TreeEnsembleClassifier(std::string(name));
        layer->init(backend::convert<float>(_base_values), backend::convert<int>(_class_ids), backend::convert<int>(_class_nodeids), backend::convert<int>(_class_treeids), backend::convert<float>(_class_weights), backend::convert<int>(_classlabels_int64s), backend::convert<std::string>(_classlabels_strings), backend::convert<int>(_nodes_falsenodeids), backend::convert<int>(_nodes_featureids), backend::convert<float>(_nodes_hitrates), backend::convert<int>(_nodes_missing_value_tracks_true), backend::convert<std::string>(_nodes_modes), backend::convert<int>(_nodes_nodeids), backend::convert<int>(_nodes_treeids), backend::convert<int>(_nodes_truenodeids), backend::convert<float>(_nodes_values), _post_transform);
        layer->bind(_X_i, _Y_o, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "TreeEnsembleClassifier" <<std::endl;

    });

    m.def("_TreeEnsembleClassifier_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: TreeEnsembleClassifier" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

void init_layer_ZipMap(py::module&);
#include "./layers/zipmap.h"
void init_layer_ZipMap(py::module& m){
    m.def("_ZipMap", [](py::str name, py::list _classlabels_int64s , py::list _classlabels_strings , py::str _X_i , py::str _Z_o) {
        layers::ZipMap* layer = new layers::ZipMap(std::string(name));
        layer->init(backend::convert<int>(_classlabels_int64s), backend::convert<std::string>(_classlabels_strings));
        layer->bind(_X_i, _Z_o);
        layer->build();
        backend::layer_dict[std::string(name)] = layer;

        //std::cout << "LAYERS ::: " << std::string(name) << " ::: " << "ZipMap" <<std::endl;

    });

    m.def("_ZipMap_run",  [](py::str name) {
        //std::cout << "RUN ::: " << std::string(name) << " ::: ZipMap" << std::endl;
        backend::layer_dict[std::string(name)]->forward();
    });

}

