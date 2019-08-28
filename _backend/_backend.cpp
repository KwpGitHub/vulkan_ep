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
#include "layers.hpp"


void create_instance() {
	backend::instance = new vuh::Instance();
	backend::device = new vuh::Device(backend::instance->devices().at(0));
	backend::file_path = "C:\\Users\\mramados.AMR";
	backend::file_path = "C:\\Users\\monish";

	backend::file_path.append("\\source\\repos\\vulkan_ep\\_backend\\");
}

void test() {
	uint32_t size = 1073741823;//1GB
	size = 536870911;//5MB
	size = 268435455;//2MB
	size = 1024;
	auto y = std::vector<float>(size, 1.0f);
	auto x = std::vector<float>(size, 2.0f);

	auto shape = std::vector<backend::Shape_t>();
	shape.push_back({ 1,1,1,1,1 });
	shape.push_back({ 1,1,1,1,1 });
	// just get the first available device
	auto device = backend::device;

	auto shape_t = vuh::Array<backend::Shape_t>(*device, shape);

	auto d_y = new backend::Tensor(y, { size,1,1,1,1 });
	auto d_x = new backend::Tensor(x, { size,1,1,1,1 });

	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
	struct Params { uint32_t size; float a; };

	vuh::Program<Specs, Params>* program;
	Params p;
	p.size = size;
	p.a = 0.1f;

	program = new vuh::Program<Specs, Params>(*device, std::string(std::string(backend::file_path) + std::string("saxpy_1.spv")).c_str());
	program->grid(size / 64, 1, 1);
	program->spec(64, 1, 1);
	program->bind(p, *d_y->data, *d_x->data, shape_t);
	program->run();

	d_y->data->toHost(begin(y));

	int error_count = 0;
	float tmp = y[0] - (1.0 + 0.1 * x[0]);
	for (int i = 0; i < size; ++i) {
		if (abs(y[i] - (1.0 + 0.1 * x[i])) > 1e-7) 
			error_count++;
	}

	if (error_count == 0)
		std::cout << ":::PIPELINE VALIDATION SUCCESS:::" << std::endl;
	else
		std::cout << ":::PIPELINE VALIDATION FAILURE:::" << std::endl;
	
	return;
}


void create_tensor(py::str name, py::array_t<float> input){
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

	//std::cout << "NP_TENSOR ::: " << name << std::endl;
	backend::Tensor* x = new backend::Tensor(data, _shape);
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
	m.def("create_layer", &create_layer);
	m.def("test", &test);

	auto nn = m.def_submodule("nn", "Neural Network C Execution");
	init_layer_LSTM(nn);
	init_layer_Identity(nn);
	init_layer_Abs(nn);
	init_layer_BatchNormalization(nn);
	init_layer_Mean(nn);
	init_layer_Add(nn);
	init_layer_GlobalMaxPool(nn);
	init_layer_Cast(nn);
	init_layer_AveragePool(nn);
	init_layer_And(nn);
	init_layer_LRN(nn);
	init_layer_ArgMax(nn);
	init_layer_Resize(nn);
	init_layer_Expand(nn);
	init_layer_Neg(nn);
	init_layer_Mul(nn);
	init_layer_ArgMin(nn);
	init_layer_CastMap(nn);
	init_layer_Exp(nn);
	init_layer_Div(nn);
	init_layer_ReverseSequence(nn);
	init_layer_Ceil(nn);
	init_layer_DepthToSpace(nn);
	init_layer_Clip(nn);
	init_layer_RNN(nn);
	init_layer_Concat(nn);
	init_layer_Constant(nn);
	init_layer_LpPool(nn);
	init_layer_Conv(nn);
	init_layer_Not(nn);
	init_layer_Gather(nn);
	init_layer_ConvTranspose(nn);
	init_layer_Dropout(nn);
	init_layer_LeakyRelu(nn);
	init_layer_Elu(nn);
	init_layer_GlobalAveragePool(nn);
	init_layer_Gemm(nn);
	init_layer_MaxPool(nn);
	init_layer_Equal(nn);
	init_layer_Tile(nn);
	init_layer_Flatten(nn);
	init_layer_Floor(nn);
	init_layer_GRU(nn);
	init_layer_GlobalLpPool(nn);
	init_layer_Greater(nn);
	init_layer_HardSigmoid(nn);
	init_layer_Selu(nn);
	init_layer_Hardmax(nn);
	init_layer_If(nn);
	init_layer_Min(nn);
	init_layer_InstanceNormalization(nn);
	init_layer_Less(nn);
	init_layer_EyeLike(nn);
	init_layer_RandomNormal(nn);
	init_layer_Slice(nn);
	init_layer_PRelu(nn);
	init_layer_Log(nn);
	init_layer_LogSoftmax(nn);
	init_layer_Loop(nn);
	init_layer_LpNormalization(nn);
	init_layer_MatMul(nn);
	init_layer_ReduceL2(nn);
	init_layer_Max(nn);
	init_layer_MaxRoiPool(nn);
	init_layer_Or(nn);
	init_layer_Pad(nn);
	init_layer_RandomUniformLike(nn);
	init_layer_Reciprocal(nn);
	init_layer_Pow(nn);
	init_layer_RandomNormalLike(nn);
	init_layer_OneHot(nn);
	init_layer_RandomUniform(nn);
	init_layer_ReduceL1(nn);
	init_layer_ReduceLogSum(nn);
	init_layer_ReduceLogSumExp(nn);
	init_layer_ReduceMax(nn);
	init_layer_OneHotEncoder(nn);
	init_layer_IsNaN(nn);
	init_layer_ReduceMean(nn);
	init_layer_ReduceMin(nn);
	init_layer_TreeEnsembleRegressor(nn);
	init_layer_ReduceProd(nn);
	init_layer_ReduceSum(nn);
	init_layer_ReduceSumSquare(nn);
	init_layer_Relu(nn);
	init_layer_Reshape(nn);
	init_layer_Shape(nn);
	init_layer_Sigmoid(nn);
	init_layer_Size(nn);
	init_layer_Softmax(nn);
	init_layer_Softplus(nn);
	init_layer_Softsign(nn);
	init_layer_SpaceToDepth(nn);
	init_layer_TfIdfVectorizer(nn);
	init_layer_Split(nn);
	init_layer_Imputer(nn);
	init_layer_Sqrt(nn);
	init_layer_Squeeze(nn);
	init_layer_TopK(nn);
	init_layer_Sub(nn);
	init_layer_Sum(nn);
	init_layer_Shrink(nn);
	init_layer_Tanh(nn);
	init_layer_Transpose(nn);
	init_layer_Unsqueeze(nn);
	init_layer_SVMClassifier(nn);
	init_layer_Xor(nn);
	init_layer_Acos(nn);
	init_layer_Asin(nn);
	init_layer_Atan(nn);
	init_layer_Cos(nn);
	init_layer_Sin(nn);
	init_layer_Tan(nn);
	init_layer_Multinomial(nn);
	init_layer_Scan(nn);
	init_layer_Compress(nn);
	init_layer_ConstantOfShape(nn);
	init_layer_MaxUnpool(nn);
	init_layer_Scatter(nn);
	init_layer_Sinh(nn);
	init_layer_Cosh(nn);
	init_layer_Asinh(nn);
	init_layer_Acosh(nn);
	init_layer_NonMaxSuppression(nn);
	init_layer_Atanh(nn);
	init_layer_Sign(nn);
	init_layer_Erf(nn);
	init_layer_Where(nn);
	init_layer_NonZero(nn);
	init_layer_MeanVarianceNormalization(nn);
	init_layer_StringNormalizer(nn);
	init_layer_Mod(nn);
	init_layer_ThresholdedRelu(nn);
	init_layer_MatMulInteger(nn);
	init_layer_QLinearMatMul(nn);
	init_layer_ConvInteger(nn);
	init_layer_QLinearConv(nn);
	init_layer_QuantizeLinear(nn);
	init_layer_DequantizeLinear(nn);
	init_layer_IsInf(nn);
	init_layer_RoiAlign(nn);
	init_layer_ArrayFeatureExtractor(nn);
	init_layer_Binarizer(nn);
	init_layer_CategoryMapper(nn);
	init_layer_DictVectorizer(nn);
	init_layer_FeatureVectorizer(nn);
	init_layer_LabelEncoder(nn);
	init_layer_LinearClassifier(nn);
	init_layer_LinearRegressor(nn);
	init_layer_Normalizer(nn);
	init_layer_SVMRegressor(nn);
	init_layer_Scaler(nn);
	init_layer_TreeEnsembleClassifier(nn);
	init_layer_ZipMap(nn);
}
