void init_layer_LSTM(py::module&);
#include "./layers/lstm.h"
void init_layer_LSTM(py::module& m){
    py::class_<backend::LSTM>(m, "_LSTM").def(py::init<const std::string&>());
        //.def("init", &backend::LSTM::init)
        //.def("bind", &backend::LSTM::bind); 
}
void init_layer_Identity(py::module&);
#include "./layers/identity.h"
void init_layer_Identity(py::module& m){
    py::class_<backend::Identity>(m, "_Identity").def(py::init<const std::string&>());
        //.def("init", &backend::Identity::init)
        //.def("bind", &backend::Identity::bind); 
}
void init_layer_Abs(py::module&);
#include "./layers/abs.h"
void init_layer_Abs(py::module& m){
    py::class_<backend::Abs>(m, "_Abs").def(py::init<const std::string&>());
        //.def("init", &backend::Abs::init)
        //.def("bind", &backend::Abs::bind); 
}
void init_layer_BatchNormalization(py::module&);
#include "./layers/batchnormalization.h"
void init_layer_BatchNormalization(py::module& m){
    py::class_<backend::BatchNormalization>(m, "_BatchNormalization").def(py::init<const std::string&>());
        //.def("init", &backend::BatchNormalization::init)
        //.def("bind", &backend::BatchNormalization::bind); 
}
void init_layer_Mean(py::module&);
#include "./layers/mean.h"
void init_layer_Mean(py::module& m){
    py::class_<backend::Mean>(m, "_Mean").def(py::init<const std::string&>());
        //.def("init", &backend::Mean::init)
        //.def("bind", &backend::Mean::bind); 
}
void init_layer_Add(py::module&);
#include "./layers/add.h"
void init_layer_Add(py::module& m){
    py::class_<backend::Add>(m, "_Add").def(py::init<const std::string&>());
        //.def("init", &backend::Add::init)
        //.def("bind", &backend::Add::bind); 
}
void init_layer_GlobalMaxPool(py::module&);
#include "./layers/globalmaxpool.h"
void init_layer_GlobalMaxPool(py::module& m){
    py::class_<backend::GlobalMaxPool>(m, "_GlobalMaxPool").def(py::init<const std::string&>());
        //.def("init", &backend::GlobalMaxPool::init)
        //.def("bind", &backend::GlobalMaxPool::bind); 
}
void init_layer_Cast(py::module&);
#include "./layers/cast.h"
void init_layer_Cast(py::module& m){
    py::class_<backend::Cast>(m, "_Cast").def(py::init<const std::string&>());
        //.def("init", &backend::Cast::init)
        //.def("bind", &backend::Cast::bind); 
}
void init_layer_AveragePool(py::module&);
#include "./layers/averagepool.h"
void init_layer_AveragePool(py::module& m){
    py::class_<backend::AveragePool>(m, "_AveragePool").def(py::init<const std::string&>());
        //.def("init", &backend::AveragePool::init)
        //.def("bind", &backend::AveragePool::bind); 
}
void init_layer_And(py::module&);
#include "./layers/and.h"
void init_layer_And(py::module& m){
    py::class_<backend::And>(m, "_And").def(py::init<const std::string&>());
        //.def("init", &backend::And::init)
        //.def("bind", &backend::And::bind); 
}
void init_layer_LRN(py::module&);
#include "./layers/lrn.h"
void init_layer_LRN(py::module& m){
    py::class_<backend::LRN>(m, "_LRN").def(py::init<const std::string&>());
        //.def("init", &backend::LRN::init)
        //.def("bind", &backend::LRN::bind); 
}
void init_layer_ArgMax(py::module&);
#include "./layers/argmax.h"
void init_layer_ArgMax(py::module& m){
    py::class_<backend::ArgMax>(m, "_ArgMax").def(py::init<const std::string&>());
        //.def("init", &backend::ArgMax::init)
        //.def("bind", &backend::ArgMax::bind); 
}
void init_layer_Resize(py::module&);
#include "./layers/resize.h"
void init_layer_Resize(py::module& m){
    py::class_<backend::Resize>(m, "_Resize").def(py::init<const std::string&>());
        //.def("init", &backend::Resize::init)
        //.def("bind", &backend::Resize::bind); 
}
void init_layer_Expand(py::module&);
#include "./layers/expand.h"
void init_layer_Expand(py::module& m){
    py::class_<backend::Expand>(m, "_Expand").def(py::init<const std::string&>());
        //.def("init", &backend::Expand::init)
        //.def("bind", &backend::Expand::bind); 
}
void init_layer_Neg(py::module&);
#include "./layers/neg.h"
void init_layer_Neg(py::module& m){
    py::class_<backend::Neg>(m, "_Neg").def(py::init<const std::string&>());
        //.def("init", &backend::Neg::init)
        //.def("bind", &backend::Neg::bind); 
}
void init_layer_Mul(py::module&);
#include "./layers/mul.h"
void init_layer_Mul(py::module& m){
    py::class_<backend::Mul>(m, "_Mul").def(py::init<const std::string&>());
        //.def("init", &backend::Mul::init)
        //.def("bind", &backend::Mul::bind); 
}
void init_layer_ArgMin(py::module&);
#include "./layers/argmin.h"
void init_layer_ArgMin(py::module& m){
    py::class_<backend::ArgMin>(m, "_ArgMin").def(py::init<const std::string&>());
        //.def("init", &backend::ArgMin::init)
        //.def("bind", &backend::ArgMin::bind); 
}
void init_layer_CastMap(py::module&);
#include "./layers/castmap.h"
void init_layer_CastMap(py::module& m){
    py::class_<backend::CastMap>(m, "_CastMap").def(py::init<const std::string&>());
        //.def("init", &backend::CastMap::init)
        //.def("bind", &backend::CastMap::bind); 
}
void init_layer_Exp(py::module&);
#include "./layers/exp.h"
void init_layer_Exp(py::module& m){
    py::class_<backend::Exp>(m, "_Exp").def(py::init<const std::string&>());
        //.def("init", &backend::Exp::init)
        //.def("bind", &backend::Exp::bind); 
}
void init_layer_Div(py::module&);
#include "./layers/div.h"
void init_layer_Div(py::module& m){
    py::class_<backend::Div>(m, "_Div").def(py::init<const std::string&>());
        //.def("init", &backend::Div::init)
        //.def("bind", &backend::Div::bind); 
}
void init_layer_ReverseSequence(py::module&);
#include "./layers/reversesequence.h"
void init_layer_ReverseSequence(py::module& m){
    py::class_<backend::ReverseSequence>(m, "_ReverseSequence").def(py::init<const std::string&>());
        //.def("init", &backend::ReverseSequence::init)
        //.def("bind", &backend::ReverseSequence::bind); 
}
void init_layer_Ceil(py::module&);
#include "./layers/ceil.h"
void init_layer_Ceil(py::module& m){
    py::class_<backend::Ceil>(m, "_Ceil").def(py::init<const std::string&>());
        //.def("init", &backend::Ceil::init)
        //.def("bind", &backend::Ceil::bind); 
}
void init_layer_DepthToSpace(py::module&);
#include "./layers/depthtospace.h"
void init_layer_DepthToSpace(py::module& m){
    py::class_<backend::DepthToSpace>(m, "_DepthToSpace").def(py::init<const std::string&>());
        //.def("init", &backend::DepthToSpace::init)
        //.def("bind", &backend::DepthToSpace::bind); 
}
void init_layer_Clip(py::module&);
#include "./layers/clip.h"
void init_layer_Clip(py::module& m){
    py::class_<backend::Clip>(m, "_Clip").def(py::init<const std::string&>());
        //.def("init", &backend::Clip::init)
        //.def("bind", &backend::Clip::bind); 
}
void init_layer_RNN(py::module&);
#include "./layers/rnn.h"
void init_layer_RNN(py::module& m){
    py::class_<backend::RNN>(m, "_RNN").def(py::init<const std::string&>());
        //.def("init", &backend::RNN::init)
        //.def("bind", &backend::RNN::bind); 
}
void init_layer_Concat(py::module&);
#include "./layers/concat.h"
void init_layer_Concat(py::module& m){
    py::class_<backend::Concat>(m, "_Concat").def(py::init<const std::string&>());
        //.def("init", &backend::Concat::init)
        //.def("bind", &backend::Concat::bind); 
}
void init_layer_Constant(py::module&);
#include "./layers/constant.h"
void init_layer_Constant(py::module& m){
    py::class_<backend::Constant>(m, "_Constant").def(py::init<const std::string&>());
        //.def("init", &backend::Constant::init)
        //.def("bind", &backend::Constant::bind); 
}
void init_layer_LpPool(py::module&);
#include "./layers/lppool.h"
void init_layer_LpPool(py::module& m){
    py::class_<backend::LpPool>(m, "_LpPool").def(py::init<const std::string&>());
        //.def("init", &backend::LpPool::init)
        //.def("bind", &backend::LpPool::bind); 
}
void init_layer_Conv(py::module&);
#include "./layers/conv.h"
void init_layer_Conv(py::module& m){
    py::class_<backend::Conv>(m, "_Conv").def(py::init<const std::string&>());
        //.def("init", &backend::Conv::init)
        //.def("bind", &backend::Conv::bind); 
}
void init_layer_Not(py::module&);
#include "./layers/not.h"
void init_layer_Not(py::module& m){
    py::class_<backend::Not>(m, "_Not").def(py::init<const std::string&>());
        //.def("init", &backend::Not::init)
        //.def("bind", &backend::Not::bind); 
}
void init_layer_Gather(py::module&);
#include "./layers/gather.h"
void init_layer_Gather(py::module& m){
    py::class_<backend::Gather>(m, "_Gather").def(py::init<const std::string&>());
        //.def("init", &backend::Gather::init)
        //.def("bind", &backend::Gather::bind); 
}
void init_layer_ConvTranspose(py::module&);
#include "./layers/convtranspose.h"
void init_layer_ConvTranspose(py::module& m){
    py::class_<backend::ConvTranspose>(m, "_ConvTranspose").def(py::init<const std::string&>());
        //.def("init", &backend::ConvTranspose::init)
        //.def("bind", &backend::ConvTranspose::bind); 
}
void init_layer_Dropout(py::module&);
#include "./layers/dropout.h"
void init_layer_Dropout(py::module& m){
    py::class_<backend::Dropout>(m, "_Dropout").def(py::init<const std::string&>());
        //.def("init", &backend::Dropout::init)
        //.def("bind", &backend::Dropout::bind); 
}
void init_layer_LeakyRelu(py::module&);
#include "./layers/leakyrelu.h"
void init_layer_LeakyRelu(py::module& m){
    py::class_<backend::LeakyRelu>(m, "_LeakyRelu").def(py::init<const std::string&>());
        //.def("init", &backend::LeakyRelu::init)
        //.def("bind", &backend::LeakyRelu::bind); 
}
void init_layer_Elu(py::module&);
#include "./layers/elu.h"
void init_layer_Elu(py::module& m){
    py::class_<backend::Elu>(m, "_Elu").def(py::init<const std::string&>());
        //.def("init", &backend::Elu::init)
        //.def("bind", &backend::Elu::bind); 
}
void init_layer_GlobalAveragePool(py::module&);
#include "./layers/globalaveragepool.h"
void init_layer_GlobalAveragePool(py::module& m){
    py::class_<backend::GlobalAveragePool>(m, "_GlobalAveragePool").def(py::init<const std::string&>());
        //.def("init", &backend::GlobalAveragePool::init)
        //.def("bind", &backend::GlobalAveragePool::bind); 
}
void init_layer_Gemm(py::module&);
#include "./layers/gemm.h"
void init_layer_Gemm(py::module& m){
    py::class_<backend::Gemm>(m, "_Gemm").def(py::init<const std::string&>());
        //.def("init", &backend::Gemm::init)
        //.def("bind", &backend::Gemm::bind); 
}
void init_layer_MaxPool(py::module&);
#include "./layers/maxpool.h"
void init_layer_MaxPool(py::module& m){
    py::class_<backend::MaxPool>(m, "_MaxPool").def(py::init<const std::string&>());
        //.def("init", &backend::MaxPool::init)
        //.def("bind", &backend::MaxPool::bind); 
}
void init_layer_Equal(py::module&);
#include "./layers/equal.h"
void init_layer_Equal(py::module& m){
    py::class_<backend::Equal>(m, "_Equal").def(py::init<const std::string&>());
        //.def("init", &backend::Equal::init)
        //.def("bind", &backend::Equal::bind); 
}
void init_layer_Tile(py::module&);
#include "./layers/tile.h"
void init_layer_Tile(py::module& m){
    py::class_<backend::Tile>(m, "_Tile").def(py::init<const std::string&>());
        //.def("init", &backend::Tile::init)
        //.def("bind", &backend::Tile::bind); 
}
void init_layer_Flatten(py::module&);
#include "./layers/flatten.h"
void init_layer_Flatten(py::module& m){
    py::class_<backend::Flatten>(m, "_Flatten").def(py::init<const std::string&>());
        //.def("init", &backend::Flatten::init)
        //.def("bind", &backend::Flatten::bind); 
}
void init_layer_Floor(py::module&);
#include "./layers/floor.h"
void init_layer_Floor(py::module& m){
    py::class_<backend::Floor>(m, "_Floor").def(py::init<const std::string&>());
        //.def("init", &backend::Floor::init)
        //.def("bind", &backend::Floor::bind); 
}
void init_layer_GRU(py::module&);
#include "./layers/gru.h"
void init_layer_GRU(py::module& m){
    py::class_<backend::GRU>(m, "_GRU").def(py::init<const std::string&>());
        //.def("init", &backend::GRU::init)
        //.def("bind", &backend::GRU::bind); 
}
void init_layer_GlobalLpPool(py::module&);
#include "./layers/globallppool.h"
void init_layer_GlobalLpPool(py::module& m){
    py::class_<backend::GlobalLpPool>(m, "_GlobalLpPool").def(py::init<const std::string&>());
        //.def("init", &backend::GlobalLpPool::init)
        //.def("bind", &backend::GlobalLpPool::bind); 
}
void init_layer_Greater(py::module&);
#include "./layers/greater.h"
void init_layer_Greater(py::module& m){
    py::class_<backend::Greater>(m, "_Greater").def(py::init<const std::string&>());
        //.def("init", &backend::Greater::init)
        //.def("bind", &backend::Greater::bind); 
}
void init_layer_HardSigmoid(py::module&);
#include "./layers/hardsigmoid.h"
void init_layer_HardSigmoid(py::module& m){
    py::class_<backend::HardSigmoid>(m, "_HardSigmoid").def(py::init<const std::string&>());
        //.def("init", &backend::HardSigmoid::init)
        //.def("bind", &backend::HardSigmoid::bind); 
}
void init_layer_Selu(py::module&);
#include "./layers/selu.h"
void init_layer_Selu(py::module& m){
    py::class_<backend::Selu>(m, "_Selu").def(py::init<const std::string&>());
        //.def("init", &backend::Selu::init)
        //.def("bind", &backend::Selu::bind); 
}
void init_layer_Hardmax(py::module&);
#include "./layers/hardmax.h"
void init_layer_Hardmax(py::module& m){
    py::class_<backend::Hardmax>(m, "_Hardmax").def(py::init<const std::string&>());
        //.def("init", &backend::Hardmax::init)
        //.def("bind", &backend::Hardmax::bind); 
}
void init_layer_If(py::module&);
#include "./layers/if.h"
void init_layer_If(py::module& m){
    py::class_<backend::If>(m, "_If").def(py::init<const std::string&>());
        //.def("init", &backend::If::init)
        //.def("bind", &backend::If::bind); 
}
void init_layer_Min(py::module&);
#include "./layers/min.h"
void init_layer_Min(py::module& m){
    py::class_<backend::Min>(m, "_Min").def(py::init<const std::string&>());
        //.def("init", &backend::Min::init)
        //.def("bind", &backend::Min::bind); 
}
void init_layer_InstanceNormalization(py::module&);
#include "./layers/instancenormalization.h"
void init_layer_InstanceNormalization(py::module& m){
    py::class_<backend::InstanceNormalization>(m, "_InstanceNormalization").def(py::init<const std::string&>());
        //.def("init", &backend::InstanceNormalization::init)
        //.def("bind", &backend::InstanceNormalization::bind); 
}
void init_layer_Less(py::module&);
#include "./layers/less.h"
void init_layer_Less(py::module& m){
    py::class_<backend::Less>(m, "_Less").def(py::init<const std::string&>());
        //.def("init", &backend::Less::init)
        //.def("bind", &backend::Less::bind); 
}
void init_layer_EyeLike(py::module&);
#include "./layers/eyelike.h"
void init_layer_EyeLike(py::module& m){
    py::class_<backend::EyeLike>(m, "_EyeLike").def(py::init<const std::string&>());
        //.def("init", &backend::EyeLike::init)
        //.def("bind", &backend::EyeLike::bind); 
}
void init_layer_RandomNormal(py::module&);
#include "./layers/randomnormal.h"
void init_layer_RandomNormal(py::module& m){
    py::class_<backend::RandomNormal>(m, "_RandomNormal").def(py::init<const std::string&>());
        //.def("init", &backend::RandomNormal::init)
        //.def("bind", &backend::RandomNormal::bind); 
}
void init_layer_Slice(py::module&);
#include "./layers/slice.h"
void init_layer_Slice(py::module& m){
    py::class_<backend::Slice>(m, "_Slice").def(py::init<const std::string&>());
        //.def("init", &backend::Slice::init)
        //.def("bind", &backend::Slice::bind); 
}
void init_layer_PRelu(py::module&);
#include "./layers/prelu.h"
void init_layer_PRelu(py::module& m){
    py::class_<backend::PRelu>(m, "_PRelu").def(py::init<const std::string&>());
        //.def("init", &backend::PRelu::init)
        //.def("bind", &backend::PRelu::bind); 
}
void init_layer_Log(py::module&);
#include "./layers/log.h"
void init_layer_Log(py::module& m){
    py::class_<backend::Log>(m, "_Log").def(py::init<const std::string&>());
        //.def("init", &backend::Log::init)
        //.def("bind", &backend::Log::bind); 
}
void init_layer_LogSoftmax(py::module&);
#include "./layers/logsoftmax.h"
void init_layer_LogSoftmax(py::module& m){
    py::class_<backend::LogSoftmax>(m, "_LogSoftmax").def(py::init<const std::string&>());
        //.def("init", &backend::LogSoftmax::init)
        //.def("bind", &backend::LogSoftmax::bind); 
}
void init_layer_Loop(py::module&);
#include "./layers/loop.h"
void init_layer_Loop(py::module& m){
    py::class_<backend::Loop>(m, "_Loop").def(py::init<const std::string&>());
        //.def("init", &backend::Loop::init)
        //.def("bind", &backend::Loop::bind); 
}
void init_layer_LpNormalization(py::module&);
#include "./layers/lpnormalization.h"
void init_layer_LpNormalization(py::module& m){
    py::class_<backend::LpNormalization>(m, "_LpNormalization").def(py::init<const std::string&>());
        //.def("init", &backend::LpNormalization::init)
        //.def("bind", &backend::LpNormalization::bind); 
}
void init_layer_MatMul(py::module&);
#include "./layers/matmul.h"
void init_layer_MatMul(py::module& m){
    py::class_<backend::MatMul>(m, "_MatMul").def(py::init<const std::string&>());
        //.def("init", &backend::MatMul::init)
        //.def("bind", &backend::MatMul::bind); 
}
void init_layer_ReduceL2(py::module&);
#include "./layers/reducel2.h"
void init_layer_ReduceL2(py::module& m){
    py::class_<backend::ReduceL2>(m, "_ReduceL2").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceL2::init)
        //.def("bind", &backend::ReduceL2::bind); 
}
void init_layer_Max(py::module&);
#include "./layers/max.h"
void init_layer_Max(py::module& m){
    py::class_<backend::Max>(m, "_Max").def(py::init<const std::string&>());
        //.def("init", &backend::Max::init)
        //.def("bind", &backend::Max::bind); 
}
void init_layer_MaxRoiPool(py::module&);
#include "./layers/maxroipool.h"
void init_layer_MaxRoiPool(py::module& m){
    py::class_<backend::MaxRoiPool>(m, "_MaxRoiPool").def(py::init<const std::string&>());
        //.def("init", &backend::MaxRoiPool::init)
        //.def("bind", &backend::MaxRoiPool::bind); 
}
void init_layer_Or(py::module&);
#include "./layers/or.h"
void init_layer_Or(py::module& m){
    py::class_<backend::Or>(m, "_Or").def(py::init<const std::string&>());
        //.def("init", &backend::Or::init)
        //.def("bind", &backend::Or::bind); 
}
void init_layer_Pad(py::module&);
#include "./layers/pad.h"
void init_layer_Pad(py::module& m){
    py::class_<backend::Pad>(m, "_Pad").def(py::init<const std::string&>());
        //.def("init", &backend::Pad::init)
        //.def("bind", &backend::Pad::bind); 
}
void init_layer_RandomUniformLike(py::module&);
#include "./layers/randomuniformlike.h"
void init_layer_RandomUniformLike(py::module& m){
    py::class_<backend::RandomUniformLike>(m, "_RandomUniformLike").def(py::init<const std::string&>());
        //.def("init", &backend::RandomUniformLike::init)
        //.def("bind", &backend::RandomUniformLike::bind); 
}
void init_layer_Reciprocal(py::module&);
#include "./layers/reciprocal.h"
void init_layer_Reciprocal(py::module& m){
    py::class_<backend::Reciprocal>(m, "_Reciprocal").def(py::init<const std::string&>());
        //.def("init", &backend::Reciprocal::init)
        //.def("bind", &backend::Reciprocal::bind); 
}
void init_layer_Pow(py::module&);
#include "./layers/pow.h"
void init_layer_Pow(py::module& m){
    py::class_<backend::Pow>(m, "_Pow").def(py::init<const std::string&>());
        //.def("init", &backend::Pow::init)
        //.def("bind", &backend::Pow::bind); 
}
void init_layer_RandomNormalLike(py::module&);
#include "./layers/randomnormallike.h"
void init_layer_RandomNormalLike(py::module& m){
    py::class_<backend::RandomNormalLike>(m, "_RandomNormalLike").def(py::init<const std::string&>());
        //.def("init", &backend::RandomNormalLike::init)
        //.def("bind", &backend::RandomNormalLike::bind); 
}
void init_layer_OneHot(py::module&);
#include "./layers/onehot.h"
void init_layer_OneHot(py::module& m){
    py::class_<backend::OneHot>(m, "_OneHot").def(py::init<const std::string&>());
        //.def("init", &backend::OneHot::init)
        //.def("bind", &backend::OneHot::bind); 
}
void init_layer_RandomUniform(py::module&);
#include "./layers/randomuniform.h"
void init_layer_RandomUniform(py::module& m){
    py::class_<backend::RandomUniform>(m, "_RandomUniform").def(py::init<const std::string&>());
        //.def("init", &backend::RandomUniform::init)
        //.def("bind", &backend::RandomUniform::bind); 
}
void init_layer_ReduceL1(py::module&);
#include "./layers/reducel1.h"
void init_layer_ReduceL1(py::module& m){
    py::class_<backend::ReduceL1>(m, "_ReduceL1").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceL1::init)
        //.def("bind", &backend::ReduceL1::bind); 
}
void init_layer_ReduceLogSum(py::module&);
#include "./layers/reducelogsum.h"
void init_layer_ReduceLogSum(py::module& m){
    py::class_<backend::ReduceLogSum>(m, "_ReduceLogSum").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceLogSum::init)
        //.def("bind", &backend::ReduceLogSum::bind); 
}
void init_layer_ReduceLogSumExp(py::module&);
#include "./layers/reducelogsumexp.h"
void init_layer_ReduceLogSumExp(py::module& m){
    py::class_<backend::ReduceLogSumExp>(m, "_ReduceLogSumExp").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceLogSumExp::init)
        //.def("bind", &backend::ReduceLogSumExp::bind); 
}
void init_layer_ReduceMax(py::module&);
#include "./layers/reducemax.h"
void init_layer_ReduceMax(py::module& m){
    py::class_<backend::ReduceMax>(m, "_ReduceMax").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceMax::init)
        //.def("bind", &backend::ReduceMax::bind); 
}
void init_layer_OneHotEncoder(py::module&);
#include "./layers/onehotencoder.h"
void init_layer_OneHotEncoder(py::module& m){
    py::class_<backend::OneHotEncoder>(m, "_OneHotEncoder").def(py::init<const std::string&>());
        //.def("init", &backend::OneHotEncoder::init)
        //.def("bind", &backend::OneHotEncoder::bind); 
}
void init_layer_IsNaN(py::module&);
#include "./layers/isnan.h"
void init_layer_IsNaN(py::module& m){
    py::class_<backend::IsNaN>(m, "_IsNaN").def(py::init<const std::string&>());
        //.def("init", &backend::IsNaN::init)
        //.def("bind", &backend::IsNaN::bind); 
}
void init_layer_ReduceMean(py::module&);
#include "./layers/reducemean.h"
void init_layer_ReduceMean(py::module& m){
    py::class_<backend::ReduceMean>(m, "_ReduceMean").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceMean::init)
        //.def("bind", &backend::ReduceMean::bind); 
}
void init_layer_ReduceMin(py::module&);
#include "./layers/reducemin.h"
void init_layer_ReduceMin(py::module& m){
    py::class_<backend::ReduceMin>(m, "_ReduceMin").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceMin::init)
        //.def("bind", &backend::ReduceMin::bind); 
}
void init_layer_TreeEnsembleRegressor(py::module&);
#include "./layers/treeensembleregressor.h"
void init_layer_TreeEnsembleRegressor(py::module& m){
    py::class_<backend::TreeEnsembleRegressor>(m, "_TreeEnsembleRegressor").def(py::init<const std::string&>());
        //.def("init", &backend::TreeEnsembleRegressor::init)
        //.def("bind", &backend::TreeEnsembleRegressor::bind); 
}
void init_layer_ReduceProd(py::module&);
#include "./layers/reduceprod.h"
void init_layer_ReduceProd(py::module& m){
    py::class_<backend::ReduceProd>(m, "_ReduceProd").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceProd::init)
        //.def("bind", &backend::ReduceProd::bind); 
}
void init_layer_ReduceSum(py::module&);
#include "./layers/reducesum.h"
void init_layer_ReduceSum(py::module& m){
    py::class_<backend::ReduceSum>(m, "_ReduceSum").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceSum::init)
        //.def("bind", &backend::ReduceSum::bind); 
}
void init_layer_ReduceSumSquare(py::module&);
#include "./layers/reducesumsquare.h"
void init_layer_ReduceSumSquare(py::module& m){
    py::class_<backend::ReduceSumSquare>(m, "_ReduceSumSquare").def(py::init<const std::string&>());
        //.def("init", &backend::ReduceSumSquare::init)
        //.def("bind", &backend::ReduceSumSquare::bind); 
}
void init_layer_Relu(py::module&);
#include "./layers/relu.h"
void init_layer_Relu(py::module& m){
    py::class_<backend::Relu>(m, "_Relu").def(py::init<const std::string&>());
        //.def("init", &backend::Relu::init)
        //.def("bind", &backend::Relu::bind); 
}
void init_layer_Reshape(py::module&);
#include "./layers/reshape.h"
void init_layer_Reshape(py::module& m){
    py::class_<backend::Reshape>(m, "_Reshape").def(py::init<const std::string&>());
        //.def("init", &backend::Reshape::init)
        //.def("bind", &backend::Reshape::bind); 
}
void init_layer_Shape(py::module&);
#include "./layers/shape.h"
void init_layer_Shape(py::module& m){
    py::class_<backend::Shape>(m, "_Shape").def(py::init<const std::string&>());
        //.def("init", &backend::Shape::init)
        //.def("bind", &backend::Shape::bind); 
}
void init_layer_Sigmoid(py::module&);
#include "./layers/sigmoid.h"
void init_layer_Sigmoid(py::module& m){
    py::class_<backend::Sigmoid>(m, "_Sigmoid").def(py::init<const std::string&>());
        //.def("init", &backend::Sigmoid::init)
        //.def("bind", &backend::Sigmoid::bind); 
}
void init_layer_Size(py::module&);
#include "./layers/size.h"
void init_layer_Size(py::module& m){
    py::class_<backend::Size>(m, "_Size").def(py::init<const std::string&>());
        //.def("init", &backend::Size::init)
        //.def("bind", &backend::Size::bind); 
}
void init_layer_Softmax(py::module&);
#include "./layers/softmax.h"
void init_layer_Softmax(py::module& m){
    py::class_<backend::Softmax>(m, "_Softmax").def(py::init<const std::string&>());
        //.def("init", &backend::Softmax::init)
        //.def("bind", &backend::Softmax::bind); 
}
void init_layer_Softplus(py::module&);
#include "./layers/softplus.h"
void init_layer_Softplus(py::module& m){
    py::class_<backend::Softplus>(m, "_Softplus").def(py::init<const std::string&>());
        //.def("init", &backend::Softplus::init)
        //.def("bind", &backend::Softplus::bind); 
}
void init_layer_Softsign(py::module&);
#include "./layers/softsign.h"
void init_layer_Softsign(py::module& m){
    py::class_<backend::Softsign>(m, "_Softsign").def(py::init<const std::string&>());
        //.def("init", &backend::Softsign::init)
        //.def("bind", &backend::Softsign::bind); 
}
void init_layer_SpaceToDepth(py::module&);
#include "./layers/spacetodepth.h"
void init_layer_SpaceToDepth(py::module& m){
    py::class_<backend::SpaceToDepth>(m, "_SpaceToDepth").def(py::init<const std::string&>());
        //.def("init", &backend::SpaceToDepth::init)
        //.def("bind", &backend::SpaceToDepth::bind); 
}
void init_layer_TfIdfVectorizer(py::module&);
#include "./layers/tfidfvectorizer.h"
void init_layer_TfIdfVectorizer(py::module& m){
    py::class_<backend::TfIdfVectorizer>(m, "_TfIdfVectorizer").def(py::init<const std::string&>());
        //.def("init", &backend::TfIdfVectorizer::init)
        //.def("bind", &backend::TfIdfVectorizer::bind); 
}
void init_layer_Split(py::module&);
#include "./layers/split.h"
void init_layer_Split(py::module& m){
    py::class_<backend::Split>(m, "_Split").def(py::init<const std::string&>());
        //.def("init", &backend::Split::init)
        //.def("bind", &backend::Split::bind); 
}
void init_layer_Imputer(py::module&);
#include "./layers/imputer.h"
void init_layer_Imputer(py::module& m){
    py::class_<backend::Imputer>(m, "_Imputer").def(py::init<const std::string&>());
        //.def("init", &backend::Imputer::init)
        //.def("bind", &backend::Imputer::bind); 
}
void init_layer_Sqrt(py::module&);
#include "./layers/sqrt.h"
void init_layer_Sqrt(py::module& m){
    py::class_<backend::Sqrt>(m, "_Sqrt").def(py::init<const std::string&>());
        //.def("init", &backend::Sqrt::init)
        //.def("bind", &backend::Sqrt::bind); 
}
void init_layer_Squeeze(py::module&);
#include "./layers/squeeze.h"
void init_layer_Squeeze(py::module& m){
    py::class_<backend::Squeeze>(m, "_Squeeze").def(py::init<const std::string&>());
        //.def("init", &backend::Squeeze::init)
        //.def("bind", &backend::Squeeze::bind); 
}
void init_layer_TopK(py::module&);
#include "./layers/topk.h"
void init_layer_TopK(py::module& m){
    py::class_<backend::TopK>(m, "_TopK").def(py::init<const std::string&>());
        //.def("init", &backend::TopK::init)
        //.def("bind", &backend::TopK::bind); 
}
void init_layer_Sub(py::module&);
#include "./layers/sub.h"
void init_layer_Sub(py::module& m){
    py::class_<backend::Sub>(m, "_Sub").def(py::init<const std::string&>());
        //.def("init", &backend::Sub::init)
        //.def("bind", &backend::Sub::bind); 
}
void init_layer_Sum(py::module&);
#include "./layers/sum.h"
void init_layer_Sum(py::module& m){
    py::class_<backend::Sum>(m, "_Sum").def(py::init<const std::string&>());
        //.def("init", &backend::Sum::init)
        //.def("bind", &backend::Sum::bind); 
}
void init_layer_Shrink(py::module&);
#include "./layers/shrink.h"
void init_layer_Shrink(py::module& m){
    py::class_<backend::Shrink>(m, "_Shrink").def(py::init<const std::string&>());
        //.def("init", &backend::Shrink::init)
        //.def("bind", &backend::Shrink::bind); 
}
void init_layer_Tanh(py::module&);
#include "./layers/tanh.h"
void init_layer_Tanh(py::module& m){
    py::class_<backend::Tanh>(m, "_Tanh").def(py::init<const std::string&>());
        //.def("init", &backend::Tanh::init)
        //.def("bind", &backend::Tanh::bind); 
}
void init_layer_Transpose(py::module&);
#include "./layers/transpose.h"
void init_layer_Transpose(py::module& m){
    py::class_<backend::Transpose>(m, "_Transpose").def(py::init<const std::string&>());
        //.def("init", &backend::Transpose::init)
        //.def("bind", &backend::Transpose::bind); 
}
void init_layer_Unsqueeze(py::module&);
#include "./layers/unsqueeze.h"
void init_layer_Unsqueeze(py::module& m){
    py::class_<backend::Unsqueeze>(m, "_Unsqueeze").def(py::init<const std::string&>());
        //.def("init", &backend::Unsqueeze::init)
        //.def("bind", &backend::Unsqueeze::bind); 
}
void init_layer_SVMClassifier(py::module&);
#include "./layers/svmclassifier.h"
void init_layer_SVMClassifier(py::module& m){
    py::class_<backend::SVMClassifier>(m, "_SVMClassifier").def(py::init<const std::string&>());
        //.def("init", &backend::SVMClassifier::init)
        //.def("bind", &backend::SVMClassifier::bind); 
}
void init_layer_Xor(py::module&);
#include "./layers/xor.h"
void init_layer_Xor(py::module& m){
    py::class_<backend::Xor>(m, "_Xor").def(py::init<const std::string&>());
        //.def("init", &backend::Xor::init)
        //.def("bind", &backend::Xor::bind); 
}
void init_layer_Acos(py::module&);
#include "./layers/acos.h"
void init_layer_Acos(py::module& m){
    py::class_<backend::Acos>(m, "_Acos").def(py::init<const std::string&>());
        //.def("init", &backend::Acos::init)
        //.def("bind", &backend::Acos::bind); 
}
void init_layer_Asin(py::module&);
#include "./layers/asin.h"
void init_layer_Asin(py::module& m){
    py::class_<backend::Asin>(m, "_Asin").def(py::init<const std::string&>());
        //.def("init", &backend::Asin::init)
        //.def("bind", &backend::Asin::bind); 
}
void init_layer_Atan(py::module&);
#include "./layers/atan.h"
void init_layer_Atan(py::module& m){
    py::class_<backend::Atan>(m, "_Atan").def(py::init<const std::string&>());
        //.def("init", &backend::Atan::init)
        //.def("bind", &backend::Atan::bind); 
}
void init_layer_Cos(py::module&);
#include "./layers/cos.h"
void init_layer_Cos(py::module& m){
    py::class_<backend::Cos>(m, "_Cos").def(py::init<const std::string&>());
        //.def("init", &backend::Cos::init)
        //.def("bind", &backend::Cos::bind); 
}
void init_layer_Sin(py::module&);
#include "./layers/sin.h"
void init_layer_Sin(py::module& m){
    py::class_<backend::Sin>(m, "_Sin").def(py::init<const std::string&>());
        //.def("init", &backend::Sin::init)
        //.def("bind", &backend::Sin::bind); 
}
void init_layer_Tan(py::module&);
#include "./layers/tan.h"
void init_layer_Tan(py::module& m){
    py::class_<backend::Tan>(m, "_Tan").def(py::init<const std::string&>());
        //.def("init", &backend::Tan::init)
        //.def("bind", &backend::Tan::bind); 
}
void init_layer_Multinomial(py::module&);
#include "./layers/multinomial.h"
void init_layer_Multinomial(py::module& m){
    py::class_<backend::Multinomial>(m, "_Multinomial").def(py::init<const std::string&>());
        //.def("init", &backend::Multinomial::init)
        //.def("bind", &backend::Multinomial::bind); 
}
void init_layer_Scan(py::module&);
#include "./layers/scan.h"
void init_layer_Scan(py::module& m){
    py::class_<backend::Scan>(m, "_Scan").def(py::init<const std::string&>());
        //.def("init", &backend::Scan::init)
        //.def("bind", &backend::Scan::bind); 
}
void init_layer_Compress(py::module&);
#include "./layers/compress.h"
void init_layer_Compress(py::module& m){
    py::class_<backend::Compress>(m, "_Compress").def(py::init<const std::string&>());
        //.def("init", &backend::Compress::init)
        //.def("bind", &backend::Compress::bind); 
}
void init_layer_ConstantOfShape(py::module&);
#include "./layers/constantofshape.h"
void init_layer_ConstantOfShape(py::module& m){
    py::class_<backend::ConstantOfShape>(m, "_ConstantOfShape").def(py::init<const std::string&>());
        //.def("init", &backend::ConstantOfShape::init)
        //.def("bind", &backend::ConstantOfShape::bind); 
}
void init_layer_MaxUnpool(py::module&);
#include "./layers/maxunpool.h"
void init_layer_MaxUnpool(py::module& m){
    py::class_<backend::MaxUnpool>(m, "_MaxUnpool").def(py::init<const std::string&>());
        //.def("init", &backend::MaxUnpool::init)
        //.def("bind", &backend::MaxUnpool::bind); 
}
void init_layer_Scatter(py::module&);
#include "./layers/scatter.h"
void init_layer_Scatter(py::module& m){
    py::class_<backend::Scatter>(m, "_Scatter").def(py::init<const std::string&>());
        //.def("init", &backend::Scatter::init)
        //.def("bind", &backend::Scatter::bind); 
}
void init_layer_Sinh(py::module&);
#include "./layers/sinh.h"
void init_layer_Sinh(py::module& m){
    py::class_<backend::Sinh>(m, "_Sinh").def(py::init<const std::string&>());
        //.def("init", &backend::Sinh::init)
        //.def("bind", &backend::Sinh::bind); 
}
void init_layer_Cosh(py::module&);
#include "./layers/cosh.h"
void init_layer_Cosh(py::module& m){
    py::class_<backend::Cosh>(m, "_Cosh").def(py::init<const std::string&>());
        //.def("init", &backend::Cosh::init)
        //.def("bind", &backend::Cosh::bind); 
}
void init_layer_Asinh(py::module&);
#include "./layers/asinh.h"
void init_layer_Asinh(py::module& m){
    py::class_<backend::Asinh>(m, "_Asinh").def(py::init<const std::string&>());
        //.def("init", &backend::Asinh::init)
        //.def("bind", &backend::Asinh::bind); 
}
void init_layer_Acosh(py::module&);
#include "./layers/acosh.h"
void init_layer_Acosh(py::module& m){
    py::class_<backend::Acosh>(m, "_Acosh").def(py::init<const std::string&>());
        //.def("init", &backend::Acosh::init)
        //.def("bind", &backend::Acosh::bind); 
}
void init_layer_NonMaxSuppression(py::module&);
#include "./layers/nonmaxsuppression.h"
void init_layer_NonMaxSuppression(py::module& m){
    py::class_<backend::NonMaxSuppression>(m, "_NonMaxSuppression").def(py::init<const std::string&>());
        //.def("init", &backend::NonMaxSuppression::init)
        //.def("bind", &backend::NonMaxSuppression::bind); 
}
void init_layer_Atanh(py::module&);
#include "./layers/atanh.h"
void init_layer_Atanh(py::module& m){
    py::class_<backend::Atanh>(m, "_Atanh").def(py::init<const std::string&>());
        //.def("init", &backend::Atanh::init)
        //.def("bind", &backend::Atanh::bind); 
}
void init_layer_Sign(py::module&);
#include "./layers/sign.h"
void init_layer_Sign(py::module& m){
    py::class_<backend::Sign>(m, "_Sign").def(py::init<const std::string&>());
        //.def("init", &backend::Sign::init)
        //.def("bind", &backend::Sign::bind); 
}
void init_layer_Erf(py::module&);
#include "./layers/erf.h"
void init_layer_Erf(py::module& m){
    py::class_<backend::Erf>(m, "_Erf").def(py::init<const std::string&>());
        //.def("init", &backend::Erf::init)
        //.def("bind", &backend::Erf::bind); 
}
void init_layer_Where(py::module&);
#include "./layers/where.h"
void init_layer_Where(py::module& m){
    py::class_<backend::Where>(m, "_Where").def(py::init<const std::string&>());
        //.def("init", &backend::Where::init)
        //.def("bind", &backend::Where::bind); 
}
void init_layer_NonZero(py::module&);
#include "./layers/nonzero.h"
void init_layer_NonZero(py::module& m){
    py::class_<backend::NonZero>(m, "_NonZero").def(py::init<const std::string&>());
        //.def("init", &backend::NonZero::init)
        //.def("bind", &backend::NonZero::bind); 
}
void init_layer_MeanVarianceNormalization(py::module&);
#include "./layers/meanvariancenormalization.h"
void init_layer_MeanVarianceNormalization(py::module& m){
    py::class_<backend::MeanVarianceNormalization>(m, "_MeanVarianceNormalization").def(py::init<const std::string&>());
        //.def("init", &backend::MeanVarianceNormalization::init)
        //.def("bind", &backend::MeanVarianceNormalization::bind); 
}
void init_layer_StringNormalizer(py::module&);
#include "./layers/stringnormalizer.h"
void init_layer_StringNormalizer(py::module& m){
    py::class_<backend::StringNormalizer>(m, "_StringNormalizer").def(py::init<const std::string&>());
        //.def("init", &backend::StringNormalizer::init)
        //.def("bind", &backend::StringNormalizer::bind); 
}
void init_layer_Mod(py::module&);
#include "./layers/mod.h"
void init_layer_Mod(py::module& m){
    py::class_<backend::Mod>(m, "_Mod").def(py::init<const std::string&>());
        //.def("init", &backend::Mod::init)
        //.def("bind", &backend::Mod::bind); 
}
void init_layer_ThresholdedRelu(py::module&);
#include "./layers/thresholdedrelu.h"
void init_layer_ThresholdedRelu(py::module& m){
    py::class_<backend::ThresholdedRelu>(m, "_ThresholdedRelu").def(py::init<const std::string&>());
        //.def("init", &backend::ThresholdedRelu::init)
        //.def("bind", &backend::ThresholdedRelu::bind); 
}
void init_layer_MatMulInteger(py::module&);
#include "./layers/matmulinteger.h"
void init_layer_MatMulInteger(py::module& m){
    py::class_<backend::MatMulInteger>(m, "_MatMulInteger").def(py::init<const std::string&>());
        //.def("init", &backend::MatMulInteger::init)
        //.def("bind", &backend::MatMulInteger::bind); 
}
void init_layer_QLinearMatMul(py::module&);
#include "./layers/qlinearmatmul.h"
void init_layer_QLinearMatMul(py::module& m){
    py::class_<backend::QLinearMatMul>(m, "_QLinearMatMul").def(py::init<const std::string&>());
        //.def("init", &backend::QLinearMatMul::init)
        //.def("bind", &backend::QLinearMatMul::bind); 
}
void init_layer_ConvInteger(py::module&);
#include "./layers/convinteger.h"
void init_layer_ConvInteger(py::module& m){
    py::class_<backend::ConvInteger>(m, "_ConvInteger").def(py::init<const std::string&>());
        //.def("init", &backend::ConvInteger::init)
        //.def("bind", &backend::ConvInteger::bind); 
}
void init_layer_QLinearConv(py::module&);
#include "./layers/qlinearconv.h"
void init_layer_QLinearConv(py::module& m){
    py::class_<backend::QLinearConv>(m, "_QLinearConv").def(py::init<const std::string&>());
        //.def("init", &backend::QLinearConv::init)
        //.def("bind", &backend::QLinearConv::bind); 
}
void init_layer_QuantizeLinear(py::module&);
#include "./layers/quantizelinear.h"
void init_layer_QuantizeLinear(py::module& m){
    py::class_<backend::QuantizeLinear>(m, "_QuantizeLinear").def(py::init<const std::string&>());
        //.def("init", &backend::QuantizeLinear::init)
        //.def("bind", &backend::QuantizeLinear::bind); 
}
void init_layer_DequantizeLinear(py::module&);
#include "./layers/dequantizelinear.h"
void init_layer_DequantizeLinear(py::module& m){
    py::class_<backend::DequantizeLinear>(m, "_DequantizeLinear").def(py::init<const std::string&>());
        //.def("init", &backend::DequantizeLinear::init)
        //.def("bind", &backend::DequantizeLinear::bind); 
}
void init_layer_IsInf(py::module&);
#include "./layers/isinf.h"
void init_layer_IsInf(py::module& m){
    py::class_<backend::IsInf>(m, "_IsInf").def(py::init<const std::string&>());
        //.def("init", &backend::IsInf::init)
        //.def("bind", &backend::IsInf::bind); 
}
void init_layer_RoiAlign(py::module&);
#include "./layers/roialign.h"
void init_layer_RoiAlign(py::module& m){
    py::class_<backend::RoiAlign>(m, "_RoiAlign").def(py::init<const std::string&>());
        //.def("init", &backend::RoiAlign::init)
        //.def("bind", &backend::RoiAlign::bind); 
}
void init_layer_ArrayFeatureExtractor(py::module&);
#include "./layers/arrayfeatureextractor.h"
void init_layer_ArrayFeatureExtractor(py::module& m){
    py::class_<backend::ArrayFeatureExtractor>(m, "_ArrayFeatureExtractor").def(py::init<const std::string&>());
        //.def("init", &backend::ArrayFeatureExtractor::init)
        //.def("bind", &backend::ArrayFeatureExtractor::bind); 
}
void init_layer_Binarizer(py::module&);
#include "./layers/binarizer.h"
void init_layer_Binarizer(py::module& m){
    py::class_<backend::Binarizer>(m, "_Binarizer").def(py::init<const std::string&>());
        //.def("init", &backend::Binarizer::init)
        //.def("bind", &backend::Binarizer::bind); 
}
void init_layer_CategoryMapper(py::module&);
#include "./layers/categorymapper.h"
void init_layer_CategoryMapper(py::module& m){
    py::class_<backend::CategoryMapper>(m, "_CategoryMapper").def(py::init<const std::string&>());
        //.def("init", &backend::CategoryMapper::init)
        //.def("bind", &backend::CategoryMapper::bind); 
}
void init_layer_DictVectorizer(py::module&);
#include "./layers/dictvectorizer.h"
void init_layer_DictVectorizer(py::module& m){
    py::class_<backend::DictVectorizer>(m, "_DictVectorizer").def(py::init<const std::string&>());
        //.def("init", &backend::DictVectorizer::init)
        //.def("bind", &backend::DictVectorizer::bind); 
}
void init_layer_FeatureVectorizer(py::module&);
#include "./layers/featurevectorizer.h"
void init_layer_FeatureVectorizer(py::module& m){
    py::class_<backend::FeatureVectorizer>(m, "_FeatureVectorizer").def(py::init<const std::string&>());
        //.def("init", &backend::FeatureVectorizer::init)
        //.def("bind", &backend::FeatureVectorizer::bind); 
}
void init_layer_LabelEncoder(py::module&);
#include "./layers/labelencoder.h"
void init_layer_LabelEncoder(py::module& m){
    py::class_<backend::LabelEncoder>(m, "_LabelEncoder").def(py::init<const std::string&>());
        //.def("init", &backend::LabelEncoder::init)
        //.def("bind", &backend::LabelEncoder::bind); 
}
void init_layer_LinearClassifier(py::module&);
#include "./layers/linearclassifier.h"
void init_layer_LinearClassifier(py::module& m){
    py::class_<backend::LinearClassifier>(m, "_LinearClassifier").def(py::init<const std::string&>());
        //.def("init", &backend::LinearClassifier::init)
        //.def("bind", &backend::LinearClassifier::bind); 
}
void init_layer_LinearRegressor(py::module&);
#include "./layers/linearregressor.h"
void init_layer_LinearRegressor(py::module& m){
    py::class_<backend::LinearRegressor>(m, "_LinearRegressor").def(py::init<const std::string&>());
        //.def("init", &backend::LinearRegressor::init)
        //.def("bind", &backend::LinearRegressor::bind); 
}
void init_layer_Normalizer(py::module&);
#include "./layers/normalizer.h"
void init_layer_Normalizer(py::module& m){
    py::class_<backend::Normalizer>(m, "_Normalizer").def(py::init<const std::string&>());
        //.def("init", &backend::Normalizer::init)
        //.def("bind", &backend::Normalizer::bind); 
}
void init_layer_SVMRegressor(py::module&);
#include "./layers/svmregressor.h"
void init_layer_SVMRegressor(py::module& m){
    py::class_<backend::SVMRegressor>(m, "_SVMRegressor").def(py::init<const std::string&>());
        //.def("init", &backend::SVMRegressor::init)
        //.def("bind", &backend::SVMRegressor::bind); 
}
void init_layer_Scaler(py::module&);
#include "./layers/scaler.h"
void init_layer_Scaler(py::module& m){
    py::class_<backend::Scaler>(m, "_Scaler").def(py::init<const std::string&>());
        //.def("init", &backend::Scaler::init)
        //.def("bind", &backend::Scaler::bind); 
}
void init_layer_TreeEnsembleClassifier(py::module&);
#include "./layers/treeensembleclassifier.h"
void init_layer_TreeEnsembleClassifier(py::module& m){
    py::class_<backend::TreeEnsembleClassifier>(m, "_TreeEnsembleClassifier").def(py::init<const std::string&>());
        //.def("init", &backend::TreeEnsembleClassifier::init)
        //.def("bind", &backend::TreeEnsembleClassifier::bind); 
}
void init_layer_ZipMap(py::module&);
#include "./layers/zipmap.h"
void init_layer_ZipMap(py::module& m){
    py::class_<backend::ZipMap>(m, "_ZipMap").def(py::init<const std::string&>());
        //.def("init", &backend::ZipMap::init)
        //.def("bind", &backend::ZipMap::bind); 
}
