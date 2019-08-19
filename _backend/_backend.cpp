#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <vector>
#include <numeric>
#include <string>


#include "kernel/vuh.h"
#include "tensor.h"
#include "layer.h"
#include "layers_map.h"

namespace backend {
	std::map<std::string, Layer*> layer_dict;
}

void test() {
	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);

	auto instance = vuh::Instance();
	auto device = instance.devices().at(0);    // just get the first available device

	auto d_y = vuh::Array<float>(device, y);   // create device arrays and copy data
	auto d_x = vuh::Array<float>(device, x);

	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	struct Params { uint32_t size; float a; };    // shader push-constants interface

	//auto program = vuh::Program<Specs, Params>(device, "C:\\Users\\monish\\source\\repos\\vulkan_ep\\_backend/saxpy.spv");
	auto program = vuh::Program<Specs, Params>(device, std::string(backend::file_path + std::string("saxpy.spv")).c_str());

	program.grid(128/64, 1, 1).spec(64, 1, 1).bind({ 128, 0.1f }, d_y, d_x); 
	program.run();
	d_y.toHost(begin(y));
	int error_count = 0;
	for (int i = 0; i < 128; ++i) {
		if (abs(y[i] - (1.0 + 0.1 * x[i])) > 1e-7) 
			error_count++;
	}

	if (error_count == 0)
		std::cout << ":::PIPELINE VALIDATION SUCCESS:::" << std::endl;
	else
		std::cout << ":::PIPELINE VALIDATION FAILURE:::" << std::endl;
	
	return;
}

void create_instance(py::str file_path) {
	std::cout << file_path << std::endl;
	backend::file_path = std::string(file_path) + std::string("/../_backend/");
	backend::instance = new vuh::Instance();
	backend::device = new vuh::Device(backend::instance->devices().at(0));
}

void create_tensor_from_numpy(py::str name, py::array_t<float> input){
	py::buffer_info buf = input.request();
	auto bs = buf.shape;
	float* p = (float*)buf.ptr;

	std::vector<float> data;
	std::vector<uint32_t> s;

	for (auto _s : bs)
		s.push_back((uint32_t)_s);

	for (int i = 0; i < std::accumulate(s.begin(), s.end(), 1, std::multiplies<uint32_t>()); ++i)
		data.push_back(p[i]);

	backend::Shape_t _shape = { 1,1,1,1,1 };

	switch (s.size()) {
	case 1: _shape = { s[0], 1,1,1,1 };
			break;
	case 2: _shape = { s[0], 1, 1, 1, s[1] };
			break;
	case 3: _shape = { s[0], 1, 1, s[1], s[2] };
			break;
	case 4: _shape = { s[0], s[1], 1, s[2], s[3] };
			break;
	case 5: _shape = { s[0], s[1], s[2], s[3], s[4] };
	}

	std::cout << "UNINIT_TENSOR ::: " << name << std::endl;
	backend::Tensor* x = new backend::Tensor(data, _shape);
	backend::tensor_dict.insert(std::pair<std::string, backend::Tensor*>(std::string(name), x));

}

void create_tensor(py::str name, py::list data, py::list shape) {
	std::vector<float> d;
	std::vector<uint32_t> s;

	for (auto x : shape)
		s.push_back(x.cast<uint32_t>());
	for (auto x : data)
		d.push_back(x.cast<float>());
	
	backend::Shape_t _shape = { 1,1,1,1,1 };

	switch (s.size()) {
	case 1: _shape = { s[0], 1,1,1,1 };
			break;
	case 2: _shape = { s[0], 1, 1, 1, s[1] };
			break;
	case 3: _shape = { s[0], 1, 1, s[1], s[2] };
			break;
	case 4: _shape = { s[0], s[1], 1, s[2], s[3] };
			break;
	case 5: _shape = { s[0], s[1], s[2], s[3], s[4] };
	}

	std::cout << "TENSOR ::: "<< name << std::endl;
	backend::Tensor* x = new backend::Tensor(d, _shape);	
	backend::tensor_dict.insert(std::pair<std::string, backend::Tensor*>(std::string(name), x));
}

void create_layer(py::str name, py::str opType, py::list inputs, py::list outputs, py::dict attribute) {
	std::vector<std::string> i;
	std::vector<std::string> o;
	std::string n = std::string(name);
	std::string oT = std::string(opType);
	std::map<std::string, std::vector<std::string>> a;

	for (auto attr : attribute) {
		auto param = std::string(py::str(attr.first));
		std::vector<std::string> tmp;
		for (auto x : attr.second) {
			tmp.push_back(std::string(py::str(x)));
		}
		a.insert(std::pair<std::string, std::vector<std::string>>(param, tmp));
	}

	for (auto x : inputs)
		i.push_back(x.cast<std::string>());

	for (auto x : outputs)
		o.push_back(x.cast<std::string>());

	std::cout << "LAYERS ::: " << name << "\n\t input:[ ";
	for (auto x : i)
		std::cout << x << " ";
	std::cout << "] \n\t output:[";
	for (auto x : o)
		std::cout << x << " ";
	std::cout << "]" << std::endl;
	
	for (auto item : attribute) {
		std::string attribute_name = std::string(py::str(item.first));
	}


}

PYBIND11_MODULE(_backend, m) {
	m.doc() = "C nn Executor";
	m.def("create_instance", &create_instance);
	m.def("create_tensor", &create_tensor);
	m.def("create_tensor_from_numpy", &create_tensor_from_numpy);
	m.def("create_layer", &create_layer);
	m.def("test", &test);

	auto nn = m.def_submodule("nn", "Neural Network C Execution");

	backend::init_layer_LSTM(nn);
	backend::init_layer_Identity(nn);
	backend::init_layer_Abs(nn);
	backend::init_layer_BatchNormalization(nn);
	backend::init_layer_Mean(nn);
	backend::init_layer_Add(nn);
	backend::init_layer_GlobalMaxPool(nn);
	backend::init_layer_Cast(nn);
	backend::init_layer_AveragePool(nn);
	backend::init_layer_And(nn);
	backend::init_layer_LRN(nn);
	backend::init_layer_ArgMax(nn);
	backend::init_layer_Resize(nn);
	backend::init_layer_Expand(nn);
	backend::init_layer_Neg(nn);
	backend::init_layer_Mul(nn);
	backend::init_layer_ArgMin(nn);
	backend::init_layer_CastMap(nn);
	backend::init_layer_Exp(nn);
	backend::init_layer_Div(nn);
	backend::init_layer_ReverseSequence(nn);
	backend::init_layer_Ceil(nn);
	backend::init_layer_DepthToSpace(nn);
	backend::init_layer_Clip(nn);
	backend::init_layer_RNN(nn);
	backend::init_layer_Concat(nn);
	backend::init_layer_Constant(nn);
	backend::init_layer_LpPool(nn);
	backend::init_layer_Conv(nn);
	backend::init_layer_Not(nn);
	backend::init_layer_Gather(nn);
	backend::init_layer_ConvTranspose(nn);
	backend::init_layer_Dropout(nn);
	backend::init_layer_LeakyRelu(nn);
	backend::init_layer_Elu(nn);
	backend::init_layer_GlobalAveragePool(nn);
	backend::init_layer_Gemm(nn);
	backend::init_layer_MaxPool(nn);
	backend::init_layer_Equal(nn);
	backend::init_layer_Tile(nn);
	backend::init_layer_Flatten(nn);
	backend::init_layer_Floor(nn);
	backend::init_layer_GRU(nn);
	backend::init_layer_GlobalLpPool(nn);
	backend::init_layer_Greater(nn);
	backend::init_layer_HardSigmoid(nn);
	backend::init_layer_Selu(nn);
	backend::init_layer_Hardmax(nn);
	backend::init_layer_If(nn);
	backend::init_layer_Min(nn);
	backend::init_layer_InstanceNormalization(nn);
	backend::init_layer_Less(nn);
	backend::init_layer_EyeLike(nn);
	backend::init_layer_RandomNormal(nn);
	backend::init_layer_Slice(nn);
	backend::init_layer_PRelu(nn);
	backend::init_layer_Log(nn);
	backend::init_layer_LogSoftmax(nn);
	backend::init_layer_Loop(nn);
	backend::init_layer_LpNormalization(nn);
	backend::init_layer_MatMul(nn);
	backend::init_layer_ReduceL2(nn);
	backend::init_layer_Max(nn);
	backend::init_layer_MaxRoiPool(nn);
	backend::init_layer_Or(nn);
	backend::init_layer_Pad(nn);
	backend::init_layer_RandomUniformLike(nn);
	backend::init_layer_Reciprocal(nn);
	backend::init_layer_Pow(nn);
	backend::init_layer_RandomNormalLike(nn);
	backend::init_layer_OneHot(nn);
	backend::init_layer_RandomUniform(nn);
	backend::init_layer_ReduceL1(nn);
	backend::init_layer_ReduceLogSum(nn);
	backend::init_layer_ReduceLogSumExp(nn);
	backend::init_layer_ReduceMax(nn);
	backend::init_layer_OneHotEncoder(nn);
	backend::init_layer_IsNaN(nn);
	backend::init_layer_ReduceMean(nn);
	backend::init_layer_ReduceMin(nn);
	backend::init_layer_TreeEnsembleRegressor(nn);
	backend::init_layer_ReduceProd(nn);
	backend::init_layer_ReduceSum(nn);
	backend::init_layer_ReduceSumSquare(nn);
	backend::init_layer_Relu(nn);
	backend::init_layer_Reshape(nn);
	backend::init_layer_Shape(nn);
	backend::init_layer_Sigmoid(nn);
	backend::init_layer_Size(nn);
	backend::init_layer_Softmax(nn);
	backend::init_layer_Softplus(nn);
	backend::init_layer_Softsign(nn);
	backend::init_layer_SpaceToDepth(nn);
	backend::init_layer_TfIdfVectorizer(nn);
	backend::init_layer_Split(nn);
	backend::init_layer_Imputer(nn);
	backend::init_layer_Sqrt(nn);
	backend::init_layer_Squeeze(nn);
	backend::init_layer_TopK(nn);
	backend::init_layer_Sub(nn);
	backend::init_layer_Sum(nn);
	backend::init_layer_Shrink(nn);
	backend::init_layer_Tanh(nn);
	backend::init_layer_Transpose(nn);
	backend::init_layer_Unsqueeze(nn);
	backend::init_layer_SVMClassifier(nn);
	backend::init_layer_Xor(nn);
	backend::init_layer_Acos(nn);
	backend::init_layer_Asin(nn);
	backend::init_layer_Atan(nn);
	backend::init_layer_Cos(nn);
	backend::init_layer_Sin(nn);
	backend::init_layer_Tan(nn);
	backend::init_layer_Multinomial(nn);
	backend::init_layer_Scan(nn);
	backend::init_layer_Compress(nn);
	backend::init_layer_ConstantOfShape(nn);
	backend::init_layer_MaxUnpool(nn);
	backend::init_layer_Scatter(nn);
	backend::init_layer_Sinh(nn);
	backend::init_layer_Cosh(nn);
	backend::init_layer_Asinh(nn);
	backend::init_layer_Acosh(nn);
	backend::init_layer_NonMaxSuppression(nn);
	backend::init_layer_Atanh(nn);
	backend::init_layer_Sign(nn);
	backend::init_layer_Erf(nn);
	backend::init_layer_Where(nn);
	backend::init_layer_NonZero(nn);
	backend::init_layer_MeanVarianceNormalization(nn);
	backend::init_layer_StringNormalizer(nn);
	backend::init_layer_Mod(nn);
	backend::init_layer_ThresholdedRelu(nn);
	backend::init_layer_MatMulInteger(nn);
	backend::init_layer_QLinearMatMul(nn);
	backend::init_layer_ConvInteger(nn);
	backend::init_layer_QLinearConv(nn);
	backend::init_layer_QuantizeLinear(nn);
	backend::init_layer_DequantizeLinear(nn);
	backend::init_layer_IsInf(nn);
	backend::init_layer_RoiAlign(nn);
	backend::init_layer_ArrayFeatureExtractor(nn);
	backend::init_layer_Binarizer(nn);
	backend::init_layer_CategoryMapper(nn);
	backend::init_layer_DictVectorizer(nn);
	backend::init_layer_FeatureVectorizer(nn);
	backend::init_layer_LabelEncoder(nn);
	backend::init_layer_LinearClassifier(nn);
	backend::init_layer_LinearRegressor(nn);
	backend::init_layer_Normalizer(nn);
	backend::init_layer_SVMRegressor(nn);
	backend::init_layer_Scaler(nn);
	backend::init_layer_TreeEnsembleClassifier(nn);
	backend::init_layer_ZipMap(nn);
}
