LSTM=
		//std::vector<float> activation_alpha;, 
		//std::vector<float> activation_beta;, 
		std::vector<std::string> activations;, 
		float clip;, 
		std::string direction;, 
		int hidden_size;, 
		int input_forget;
Identity=
Abs=
BatchNormalization=
		float epsilon;, 
		float momentum;
Mean=
Add=
GlobalMaxPool=
Cast=
		int to;
AveragePool=
		std::string auto_pad;, 
		int ceil_mode;, 
		int count_include_pad;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
And=
LRN=
		float alpha;, 
		float beta;, 
		float bias;, 
		int size;
ArgMax=
		int axis;, 
		int keepdims;
Resize=
		std::string mode;
Expand=
Neg=
Mul=
ArgMin=
		int axis;, 
		int keepdims;
CastMap=
		std::string cast_to;, 
		std::string map_form;, 
		int max_map;
Exp=
Div=
ReverseSequence=
		int batch_axis;, 
		int time_axis;
Ceil=
DepthToSpace=
		int blocksize;
Clip=
		float max;, 
		float min;
RNN=
		//std::vector<float> activation_alpha;, 
		//std::vector<float> activation_beta;, 
		std::vector<std::string> activations;, 
		float clip;, 
		std::string direction;, 
		int hidden_size;
Concat=
		int axis;
Constant=
		//tensor value;
LpPool=
		std::string auto_pad;, 
		std::vector<int> kernel_shape;, 
		int p;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
Conv=
		std::string auto_pad;, 
		std::vector<int> dilations;, 
		int group;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
Not=
Gather=
		int axis;
ConvTranspose=
		std::string auto_pad;, 
		std::vector<int> dilations;, 
		int group;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> output_padding;, 
		std::vector<int> output_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
Dropout=
		float ratio;
LeakyRelu=
		float alpha;
Elu=
		float alpha;
GlobalAveragePool=
Gemm=
		float alpha;, 
		float beta;, 
		int transA;, 
		int transB;
MaxPool=
		std::string auto_pad;, 
		int ceil_mode;, 
		std::vector<int> dilations;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		int storage_order;, 
		std::vector<int> strides;
Equal=
Tile=
Flatten=
		int axis;
Floor=
GRU=
		//std::vector<float> activation_alpha;, 
		//std::vector<float> activation_beta;, 
		std::vector<std::string> activations;, 
		float clip;, 
		std::string direction;, 
		int hidden_size;, 
		int linear_before_reset;
GlobalLpPool=
		int p;
Greater=
HardSigmoid=
		float alpha;, 
		float beta;
Selu=
		float alpha;, 
		float gamma;
Hardmax=
		int axis;
If=
		//graph else_branch;, 
		//graph then_branch;
Min=
InstanceNormalization=
		float epsilon;
Less=
EyeLike=
		int dtype;, 
		int k;
RandomNormal=
		int dtype;, 
		float mean;, 
		float scale;, 
		float seed;, 
		std::vector<int> shape;
Slice=
PRelu=
Log=
LogSoftmax=
		int axis;
Loop=
		//graph body;
LpNormalization=
		int axis;, 
		int p;
MatMul=
ReduceL2=
		std::vector<int> axes;, 
		int keepdims;
Max=
MaxRoiPool=
		std::vector<int> pooled_shape;, 
		float spatial_scale;
Or=
Pad=
		std::string mode;, 
		std::vector<int> pads;, 
		float value;
RandomUniformLike=
		int dtype;, 
		float high;, 
		float low;, 
		float seed;
Reciprocal=
Pow=
RandomNormalLike=
		int dtype;, 
		float mean;, 
		float scale;, 
		float seed;
OneHot=
		int axis;
RandomUniform=
		int dtype;, 
		float high;, 
		float low;, 
		float seed;, 
		std::vector<int> shape;
ReduceL1=
		std::vector<int> axes;, 
		int keepdims;
ReduceLogSum=
		std::vector<int> axes;, 
		int keepdims;
ReduceLogSumExp=
		std::vector<int> axes;, 
		int keepdims;
ReduceMax=
		std::vector<int> axes;, 
		int keepdims;
OneHotEncoder=
		std::vector<int> cats_int64s;, 
		std::vector<std::string> cats_strings;, 
		int zeros;
IsNaN=
ReduceMean=
		std::vector<int> axes;, 
		int keepdims;
ReduceMin=
		std::vector<int> axes;, 
		int keepdims;
TreeEnsembleRegressor=
		std::string aggregate_function;, 
		//std::vector<float> base_values;, 
		int n_targets;, 
		std::vector<int> nodes_falsenodeids;, 
		std::vector<int> nodes_featureids;, 
		//std::vector<float> nodes_hitrates;, 
		std::vector<int> nodes_missing_value_tracks_true;, 
		std::vector<std::string> nodes_modes;, 
		std::vector<int> nodes_nodeids;, 
		std::vector<int> nodes_treeids;, 
		std::vector<int> nodes_truenodeids;, 
		//std::vector<float> nodes_values;, 
		std::string post_transform;, 
		std::vector<int> target_ids;, 
		std::vector<int> target_nodeids;, 
		std::vector<int> target_treeids;, 
		//std::vector<float> target_weights;
ReduceProd=
		std::vector<int> axes;, 
		int keepdims;
ReduceSum=
		std::vector<int> axes;, 
		int keepdims;
ReduceSumSquare=
		std::vector<int> axes;, 
		int keepdims;
Relu=
Reshape=
Shape=
Sigmoid=
Size=
Softmax=
		int axis;
Softplus=
Softsign=
SpaceToDepth=
		int blocksize;
TfIdfVectorizer=
		int max_gram_length;, 
		int max_skip_count;, 
		int min_gram_length;, 
		std::string mode;, 
		std::vector<int> ngram_counts;, 
		std::vector<int> ngram_indexes;, 
		std::vector<int> pool_int64s;, 
		std::vector<std::string> pool_strings;, 
		//std::vector<float> weights;
Split=
		int axis;, 
		std::vector<int> split;
Imputer=
		//std::vector<float> imputed_value_floats;, 
		std::vector<int> imputed_value_int64s;, 
		float replaced_value_float;, 
		int replaced_value_int64;
Sqrt=
Squeeze=
		std::vector<int> axes;
TopK=
		int axis;
Sub=
Sum=
Shrink=
		float bias;, 
		float lambd;
Tanh=
Transpose=
		std::vector<int> perm;
Unsqueeze=
		std::vector<int> axes;
Upsample=
		std::string mode;
SVMClassifier=
		std::vector<int> classlabels_ints;, 
		std::vector<std::string> classlabels_strings;, 
		//std::vector<float> coefficients;, 
		//std::vector<float> kernel_params;, 
		std::string kernel_type;, 
		std::string post_transform;, 
		//std::vector<float> prob_a;, 
		//std::vector<float> prob_b;, 
		//std::vector<float> rho;, 
		//std::vector<float> support_vectors;, 
		std::vector<int> vectors_per_class;
Xor=
Acos=
Asin=
Atan=
Cos=
Sin=
Tan=
Multinomial=
		int dtype;, 
		int sample_size;, 
		float seed;
Scan=
		//graph body;, 
		int num_scan_inputs;, 
		std::vector<int> scan_input_axes;, 
		std::vector<int> scan_input_directions;, 
		std::vector<int> scan_output_axes;, 
		std::vector<int> scan_output_directions;
Compress=
		int axis;
ConstantOfShape=
		//tensor value;
MaxUnpool=
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
Scatter=
		int axis;
Sinh=
Cosh=
Asinh=
Acosh=
NonMaxSuppression=
		int center_point_box;
Atanh=
Sign=
Erf=
Where=
NonZero=
MeanVarianceNormalization=
		std::vector<int> axes;
StringNormalizer=
		std::string case_change_action;, 
		int is_case_sensitive;, 
		std::string locale;, 
		std::vector<std::string> stopwords;
Mod=
		int fmod;
ThresholdedRelu=
		float alpha;
MatMulInteger=
QLinearMatMul=
ConvInteger=
		std::string auto_pad;, 
		std::vector<int> dilations;, 
		int group;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
QLinearConv=
		std::string auto_pad;, 
		std::vector<int> dilations;, 
		int group;, 
		std::vector<int> kernel_shape;, 
		std::vector<int> pads;, 
		std::vector<int> strides;
QuantizeLinear=
DequantizeLinear=
IsInf=
		int detect_negative;, 
		int detect_positive;
RoiAlign=
		std::string mode;, 
		int output_height;, 
		int output_width;, 
		int sampling_ratio;, 
		float spatial_scale;
ArrayFeatureExtractor=
Binarizer=
		float threshold;
CategoryMapper=
		std::vector<int> cats_int64s;, 
		std::vector<std::string> cats_strings;, 
		int default_int64;, 
		std::string default_string;
DictVectorizer=
		std::vector<int> int64_vocabulary;, 
		std::vector<std::string> string_vocabulary;
FeatureVectorizer=
		std::vector<int> inputdimensions;
LabelEncoder=
		float default_float;, 
		int default_int64;, 
		std::string default_string;, 
		//std::vector<float> keys_floats;, 
		std::vector<int> keys_int64s;, 
		std::vector<std::string> keys_strings;, 
		//std::vector<float> values_floats;, 
		std::vector<int> values_int64s;, 
		std::vector<std::string> values_strings;
LinearClassifier=
		std::vector<int> classlabels_ints;, 
		std::vector<std::string> classlabels_strings;, 
		//std::vector<float> coefficients;, 
		//std::vector<float> intercepts;, 
		int multi_class;, 
		std::string post_transform;
LinearRegressor=
		//std::vector<float> coefficients;, 
		//std::vector<float> intercepts;, 
		std::string post_transform;, 
		int targets;
Normalizer=
		std::string norm;
SVMRegressor=
		//std::vector<float> coefficients;, 
		//std::vector<float> kernel_params;, 
		std::string kernel_type;, 
		int n_supports;, 
		int one_class;, 
		std::string post_transform;, 
		//std::vector<float> rho;, 
		//std::vector<float> support_vectors;
Scaler=
		//std::vector<float> offset;, 
		//std::vector<float> scale;
TreeEnsembleClassifier=
		//std::vector<float> base_values;, 
		std::vector<int> class_ids;, 
		std::vector<int> class_nodeids;, 
		std::vector<int> class_treeids;, 
		//std::vector<float> class_weights;, 
		std::vector<int> classlabels_int64s;, 
		std::vector<std::string> classlabels_strings;, 
		std::vector<int> nodes_falsenodeids;, 
		std::vector<int> nodes_featureids;, 
		//std::vector<float> nodes_hitrates;, 
		std::vector<int> nodes_missing_value_tracks_true;, 
		std::vector<std::string> nodes_modes;, 
		std::vector<int> nodes_nodeids;, 
		std::vector<int> nodes_treeids;, 
		std::vector<int> nodes_truenodeids;, 
		//std::vector<float> nodes_values;, 
		std::string post_transform;
ZipMap=
		std::vector<int> classlabels_int64s;, 
		std::vector<std::string> classlabels_strings;
