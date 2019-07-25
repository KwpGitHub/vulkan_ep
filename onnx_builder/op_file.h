LSTM=
		float[] activation_alpha;, 
		float[] activation_beta;, 
		std::string[] activations;, 
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
		int[] kernel_shape;, 
		int[] pads;, 
		int[] strides;
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
		float[] activation_alpha;, 
		float[] activation_beta;, 
		std::string[] activations;, 
		float clip;, 
		std::string direction;, 
		int hidden_size;
Concat=
		int axis;
Constant=
		//tensor value;
LpPool=
		std::string auto_pad;, 
		int[] kernel_shape;, 
		int p;, 
		int[] pads;, 
		int[] strides;
Conv=
		std::string auto_pad;, 
		int[] dilations;, 
		int group;, 
		int[] kernel_shape;, 
		int[] pads;, 
		int[] strides;
Not=
Gather=
		int axis;
ConvTranspose=
		std::string auto_pad;, 
		int[] dilations;, 
		int group;, 
		int[] kernel_shape;, 
		int[] output_padding;, 
		int[] output_shape;, 
		int[] pads;, 
		int[] strides;
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
		int[] dilations;, 
		int[] kernel_shape;, 
		int[] pads;, 
		int storage_order;, 
		int[] strides;
Equal=
Tile=
Flatten=
		int axis;
Floor=
GRU=
		float[] activation_alpha;, 
		float[] activation_beta;, 
		std::string[] activations;, 
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
		int[] shape;
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
		int[] axes;, 
		int keepdims;
Max=
MaxRoiPool=
		int[] pooled_shape;, 
		float spatial_scale;
Or=
Pad=
		std::string mode;, 
		int[] pads;, 
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
		int[] shape;
ReduceL1=
		int[] axes;, 
		int keepdims;
ReduceLogSum=
		int[] axes;, 
		int keepdims;
ReduceLogSumExp=
		int[] axes;, 
		int keepdims;
ReduceMax=
		int[] axes;, 
		int keepdims;
OneHotEncoder=
		int[] cats_int64s;, 
		std::string[] cats_strings;, 
		int zeros;
IsNaN=
ReduceMean=
		int[] axes;, 
		int keepdims;
ReduceMin=
		int[] axes;, 
		int keepdims;
TreeEnsembleRegressor=
		std::string aggregate_function;, 
		float[] base_values;, 
		int n_targets;, 
		int[] nodes_falsenodeids;, 
		int[] nodes_featureids;, 
		float[] nodes_hitrates;, 
		int[] nodes_missing_value_tracks_true;, 
		std::string[] nodes_modes;, 
		int[] nodes_nodeids;, 
		int[] nodes_treeids;, 
		int[] nodes_truenodeids;, 
		float[] nodes_values;, 
		std::string post_transform;, 
		int[] target_ids;, 
		int[] target_nodeids;, 
		int[] target_treeids;, 
		float[] target_weights;
ReduceProd=
		int[] axes;, 
		int keepdims;
ReduceSum=
		int[] axes;, 
		int keepdims;
ReduceSumSquare=
		int[] axes;, 
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
		int[] ngram_counts;, 
		int[] ngram_indexes;, 
		int[] pool_int64s;, 
		std::string[] pool_strings;, 
		float[] weights;
Split=
		int axis;, 
		int[] split;
Imputer=
		float[] imputed_value_floats;, 
		int[] imputed_value_int64s;, 
		float replaced_value_float;, 
		int replaced_value_int64;
Sqrt=
Squeeze=
		int[] axes;
TopK=
		int axis;
Sub=
Sum=
Shrink=
		float bias;, 
		float lambd;
Tanh=
Transpose=
		int[] perm;
Unsqueeze=
		int[] axes;
Upsample=
		std::string mode;
SVMClassifier=
		int[] classlabels_ints;, 
		std::string[] classlabels_strings;, 
		float[] coefficients;, 
		float[] kernel_params;, 
		std::string kernel_type;, 
		std::string post_transform;, 
		float[] prob_a;, 
		float[] prob_b;, 
		float[] rho;, 
		float[] support_vectors;, 
		int[] vectors_per_class;
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
		int[] scan_input_axes;, 
		int[] scan_input_directions;, 
		int[] scan_output_axes;, 
		int[] scan_output_directions;
Compress=
		int axis;
ConstantOfShape=
		//tensor value;
MaxUnpool=
		int[] kernel_shape;, 
		int[] pads;, 
		int[] strides;
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
		int[] axes;
StringNormalizer=
		std::string case_change_action;, 
		int is_case_sensitive;, 
		std::string locale;, 
		std::string[] stopwords;
Mod=
		int fmod;
ThresholdedRelu=
		float alpha;
MatMulInteger=
QLinearMatMul=
ConvInteger=
		std::string auto_pad;, 
		int[] dilations;, 
		int group;, 
		int[] kernel_shape;, 
		int[] pads;, 
		int[] strides;
QLinearConv=
		std::string auto_pad;, 
		int[] dilations;, 
		int group;, 
		int[] kernel_shape;, 
		int[] pads;, 
		int[] strides;
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
		int[] cats_int64s;, 
		std::string[] cats_strings;, 
		int default_int64;, 
		std::string default_string;
DictVectorizer=
		int[] int64_vocabulary;, 
		std::string[] string_vocabulary;
FeatureVectorizer=
		int[] inputdimensions;
LabelEncoder=
		float default_float;, 
		int default_int64;, 
		std::string default_string;, 
		float[] keys_floats;, 
		int[] keys_int64s;, 
		std::string[] keys_strings;, 
		float[] values_floats;, 
		int[] values_int64s;, 
		std::string[] values_strings;
LinearClassifier=
		int[] classlabels_ints;, 
		std::string[] classlabels_strings;, 
		float[] coefficients;, 
		float[] intercepts;, 
		int multi_class;, 
		std::string post_transform;
LinearRegressor=
		float[] coefficients;, 
		float[] intercepts;, 
		std::string post_transform;, 
		int targets;
Normalizer=
		std::string norm;
SVMRegressor=
		float[] coefficients;, 
		float[] kernel_params;, 
		std::string kernel_type;, 
		int n_supports;, 
		int one_class;, 
		std::string post_transform;, 
		float[] rho;, 
		float[] support_vectors;
Scaler=
		float[] offset;, 
		float[] scale;
TreeEnsembleClassifier=
		float[] base_values;, 
		int[] class_ids;, 
		int[] class_nodeids;, 
		int[] class_treeids;, 
		float[] class_weights;, 
		int[] classlabels_int64s;, 
		std::string[] classlabels_strings;, 
		int[] nodes_falsenodeids;, 
		int[] nodes_featureids;, 
		float[] nodes_hitrates;, 
		int[] nodes_missing_value_tracks_true;, 
		std::string[] nodes_modes;, 
		int[] nodes_nodeids;, 
		int[] nodes_treeids;, 
		int[] nodes_truenodeids;, 
		float[] nodes_values;, 
		std::string post_transform;
ZipMap=
		int[] classlabels_int64s;, 
		std::string[] classlabels_strings;
