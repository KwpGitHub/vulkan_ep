LSTM=
			Shape_t X;, 
			Shape_t W;, 
			Shape_t R;, 
			Shape_t B;, 
			Shape_t sequence_lens;, 
			Shape_t initial_h;, 
			Shape_t initial_c;, 
			Shape_t P;, 
			Shape_t Y;, 
			Shape_t Y_h;, 
			Shape_t Y_c;, 
			float* activation_alpha;, 
			float* activation_beta;, 
			std::vector<std::string> activations;, 
			float clip;, 
			int direction;, 
			int hidden_size;, 
			int input_forget;
Identity=
			Shape_t input;, 
			Shape_t output;
Abs=
			Shape_t X;, 
			Shape_t Y;
BatchNormalization=
			Shape_t X;, 
			Shape_t scale;, 
			Shape_t B;, 
			Shape_t mean;, 
			Shape_t var;, 
			Shape_t Y;, 
			Shape_t mean;, 
			Shape_t var;, 
			Shape_t saved_mean;, 
			Shape_t saved_var;, 
			float epsilon;, 
			float momentum;
Mean=
			Shape_t data_0;, 
			Shape_t mean;
Add=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
GlobalMaxPool=
			Shape_t X;, 
			Shape_t Y;
Cast=
			Shape_t input;, 
			Shape_t output;, 
			int to;
AveragePool=
			Shape_t X;, 
			Shape_t Y;, 
			int auto_pad;, 
			int ceil_mode;, 
			int count_include_pad;, 
			int* kernel_shape;, 
			int* pads;, 
			int* strides;
And=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
LRN=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;, 
			float beta;, 
			float bias;, 
			int size;
ArgMax=
			Shape_t data;, 
			Shape_t reduced;, 
			int axis;, 
			int keepdims;
Resize=
			Shape_t X;, 
			Shape_t scales;, 
			Shape_t Y;, 
			int mode;
Expand=
			Shape_t input;, 
			Shape_t shape;, 
			Shape_t output;
Neg=
			Shape_t X;, 
			Shape_t Y;
Mul=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
ArgMin=
			Shape_t data;, 
			Shape_t reduced;, 
			int axis;, 
			int keepdims;
CastMap=
			Shape_t X;, 
			Shape_t Y;, 
			int cast_to;, 
			int map_form;, 
			int max_map;
Exp=
			Shape_t input;, 
			Shape_t output;
Div=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
ReverseSequence=
			Shape_t input;, 
			Shape_t sequence_lens;, 
			Shape_t Y;, 
			int batch_axis;, 
			int time_axis;
Ceil=
			Shape_t X;, 
			Shape_t Y;
DepthToSpace=
			Shape_t input;, 
			Shape_t output;, 
			int blocksize;
Clip=
			Shape_t input;, 
			Shape_t output;, 
			float max;, 
			float min;
RNN=
			Shape_t X;, 
			Shape_t W;, 
			Shape_t R;, 
			Shape_t B;, 
			Shape_t sequence_lens;, 
			Shape_t initial_h;, 
			Shape_t Y;, 
			Shape_t Y_h;, 
			float* activation_alpha;, 
			float* activation_beta;, 
			std::vector<std::string> activations;, 
			float clip;, 
			int direction;, 
			int hidden_size;
Concat=
			Shape_t inputs;, 
			Shape_t concat_result;, 
			int axis;
Constant=
			Shape_t output;, 
			//tensor value;
LpPool=
			Shape_t X;, 
			Shape_t Y;, 
			int auto_pad;, 
			int* kernel_shape;, 
			int p;, 
			int* pads;, 
			int* strides;
Conv=
			Shape_t X;, 
			Shape_t W;, 
			Shape_t B;, 
			Shape_t Y;, 
			int auto_pad;, 
			int* dilations;, 
			int group;, 
			int* kernel_shape;, 
			int* pads;, 
			int* strides;
Not=
			Shape_t X;, 
			Shape_t Y;
Gather=
			Shape_t data;, 
			Shape_t indices;, 
			Shape_t output;, 
			int axis;
ConvTranspose=
			Shape_t X;, 
			Shape_t W;, 
			Shape_t B;, 
			Shape_t Y;, 
			int auto_pad;, 
			int* dilations;, 
			int group;, 
			int* kernel_shape;, 
			int* output_padding;, 
			int* output_shape;, 
			int* pads;, 
			int* strides;
Dropout=
			Shape_t data;, 
			Shape_t output;, 
			Shape_t mask;, 
			float ratio;
LeakyRelu=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;
Elu=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;
GlobalAveragePool=
			Shape_t X;, 
			Shape_t Y;
Gemm=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;, 
			Shape_t Y;, 
			float alpha;, 
			float beta;, 
			int transA;, 
			int transB;
MaxPool=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Indices;, 
			int auto_pad;, 
			int ceil_mode;, 
			int* dilations;, 
			int* kernel_shape;, 
			int* pads;, 
			int storage_order;, 
			int* strides;
Equal=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
Tile=
			Shape_t input;, 
			Shape_t repeats;, 
			Shape_t output;
Flatten=
			Shape_t input;, 
			Shape_t output;, 
			int axis;
Floor=
			Shape_t X;, 
			Shape_t Y;
GRU=
			Shape_t X;, 
			Shape_t W;, 
			Shape_t R;, 
			Shape_t B;, 
			Shape_t sequence_lens;, 
			Shape_t initial_h;, 
			Shape_t Y;, 
			Shape_t Y_h;, 
			float* activation_alpha;, 
			float* activation_beta;, 
			std::vector<std::string> activations;, 
			float clip;, 
			int direction;, 
			int hidden_size;, 
			int linear_before_reset;
GlobalLpPool=
			Shape_t X;, 
			Shape_t Y;, 
			int p;
Greater=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
HardSigmoid=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;, 
			float beta;
Selu=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;, 
			float gamma;
Hardmax=
			Shape_t input;, 
			Shape_t output;, 
			int axis;
If=
			Shape_t cond;, 
			Shape_t outputs;, 
			//graph else_branch;, 
			//graph then_branch;
Min=
			Shape_t data_0;, 
			Shape_t min;
InstanceNormalization=
			Shape_t input;, 
			Shape_t scale;, 
			Shape_t B;, 
			Shape_t output;, 
			float epsilon;
Less=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
EyeLike=
			Shape_t input;, 
			Shape_t output;, 
			int dtype;, 
			int k;
RandomNormal=
			Shape_t output;, 
			int dtype;, 
			float mean;, 
			float scale;, 
			float seed;, 
			int* shape;
Slice=
			Shape_t data;, 
			Shape_t starts;, 
			Shape_t ends;, 
			Shape_t axes;, 
			Shape_t steps;, 
			Shape_t output;
PRelu=
			Shape_t X;, 
			Shape_t slope;, 
			Shape_t Y;
Log=
			Shape_t input;, 
			Shape_t output;
LogSoftmax=
			Shape_t input;, 
			Shape_t output;, 
			int axis;
Loop=
			Shape_t M;, 
			Shape_t cond;, 
			Shape_t v_initial;, 
			Shape_t v_final_and_scan_outputs;, 
			//graph body;
LpNormalization=
			Shape_t input;, 
			Shape_t output;, 
			int axis;, 
			int p;
MatMul=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t Y;
ReduceL2=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
Max=
			Shape_t data_0;, 
			Shape_t max;
MaxRoiPool=
			Shape_t X;, 
			Shape_t rois;, 
			Shape_t Y;, 
			int* pooled_shape;, 
			float spatial_scale;
Or=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
Pad=
			Shape_t data;, 
			Shape_t output;, 
			int mode;, 
			int* pads;, 
			float value;
RandomUniformLike=
			Shape_t input;, 
			Shape_t output;, 
			int dtype;, 
			float high;, 
			float low;, 
			float seed;
Reciprocal=
			Shape_t X;, 
			Shape_t Y;
Pow=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Z;
RandomNormalLike=
			Shape_t input;, 
			Shape_t output;, 
			int dtype;, 
			float mean;, 
			float scale;, 
			float seed;
OneHot=
			Shape_t indices;, 
			Shape_t depth;, 
			Shape_t values;, 
			Shape_t output;, 
			int axis;
RandomUniform=
			Shape_t output;, 
			int dtype;, 
			float high;, 
			float low;, 
			float seed;, 
			int* shape;
ReduceL1=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceLogSum=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceLogSumExp=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceMax=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
OneHotEncoder=
			Shape_t X;, 
			Shape_t Y;, 
			int* cats_int64s;, 
			std::vector<std::string> cats_strings;, 
			int zeros;
IsNaN=
			Shape_t X;, 
			Shape_t Y;
ReduceMean=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceMin=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
TreeEnsembleRegressor=
			Shape_t X;, 
			Shape_t Y;, 
			int aggregate_function;, 
			float* base_values;, 
			int n_targets;, 
			int* nodes_falsenodeids;, 
			int* nodes_featureids;, 
			float* nodes_hitrates;, 
			int* nodes_missing_value_tracks_true;, 
			std::vector<std::string> nodes_modes;, 
			int* nodes_nodeids;, 
			int* nodes_treeids;, 
			int* nodes_truenodeids;, 
			float* nodes_values;, 
			int post_transform;, 
			int* target_ids;, 
			int* target_nodeids;, 
			int* target_treeids;, 
			float* target_weights;
ReduceProd=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceSum=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
ReduceSumSquare=
			Shape_t data;, 
			Shape_t reduced;, 
			int* axes;, 
			int keepdims;
Relu=
			Shape_t X;, 
			Shape_t Y;
Reshape=
			Shape_t data;, 
			Shape_t shape;, 
			Shape_t reshaped;
Shape=
			Shape_t data;, 
			Shape_t shape;
Sigmoid=
			Shape_t X;, 
			Shape_t Y;
Size=
			Shape_t data;, 
			Shape_t size;
Softmax=
			Shape_t input;, 
			Shape_t output;, 
			int axis;
Softplus=
			Shape_t X;, 
			Shape_t Y;
Softsign=
			Shape_t input;, 
			Shape_t output;
SpaceToDepth=
			Shape_t input;, 
			Shape_t output;, 
			int blocksize;
TfIdfVectorizer=
			Shape_t X;, 
			Shape_t Y;, 
			int max_gram_length;, 
			int max_skip_count;, 
			int min_gram_length;, 
			int mode;, 
			int* ngram_counts;, 
			int* ngram_indexes;, 
			int* pool_int64s;, 
			std::vector<std::string> pool_strings;, 
			float* weights;
Split=
			Shape_t input;, 
			Shape_t outputs;, 
			int axis;, 
			int* split;
Imputer=
			Shape_t X;, 
			Shape_t Y;, 
			float* imputed_value_floats;, 
			int* imputed_value_int64s;, 
			float replaced_value_float;, 
			int replaced_value_int64;
Sqrt=
			Shape_t X;, 
			Shape_t Y;
Squeeze=
			Shape_t data;, 
			Shape_t squeezed;, 
			int* axes;
TopK=
			Shape_t X;, 
			Shape_t K;, 
			Shape_t Values;, 
			Shape_t Indices;, 
			int axis;
Sub=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
Sum=
			Shape_t data_0;, 
			Shape_t sum;
Shrink=
			Shape_t input;, 
			Shape_t output;, 
			float bias;, 
			float lambd;
Tanh=
			Shape_t input;, 
			Shape_t output;
Transpose=
			Shape_t data;, 
			Shape_t transposed;, 
			int* perm;
Unsqueeze=
			Shape_t data;, 
			Shape_t expanded;, 
			int* axes;
Upsample=
			Shape_t X;, 
			Shape_t scales;, 
			Shape_t Y;, 
			int mode;
SVMClassifier=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Z;, 
			int* classlabels_ints;, 
			std::vector<std::string> classlabels_strings;, 
			float* coefficients;, 
			float* kernel_params;, 
			int kernel_type;, 
			int post_transform;, 
			float* prob_a;, 
			float* prob_b;, 
			float* rho;, 
			float* support_vectors;, 
			int* vectors_per_class;
Xor=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;
Acos=
			Shape_t input;, 
			Shape_t output;
Asin=
			Shape_t input;, 
			Shape_t output;
Atan=
			Shape_t input;, 
			Shape_t output;
Cos=
			Shape_t input;, 
			Shape_t output;
Sin=
			Shape_t input;, 
			Shape_t output;
Tan=
			Shape_t input;, 
			Shape_t output;
Multinomial=
			Shape_t input;, 
			Shape_t output;, 
			int dtype;, 
			int sample_size;, 
			float seed;
Scan=
			Shape_t initial_state_and_scan_inputs;, 
			Shape_t final_state_and_scan_outputs;, 
			//graph body;, 
			int num_scan_inputs;, 
			int* scan_input_axes;, 
			int* scan_input_directions;, 
			int* scan_output_axes;, 
			int* scan_output_directions;
Compress=
			Shape_t input;, 
			Shape_t condition;, 
			Shape_t output;, 
			int axis;
ConstantOfShape=
			Shape_t input;, 
			Shape_t output;, 
			//tensor value;
MaxUnpool=
			Shape_t X;, 
			Shape_t I;, 
			Shape_t output_shape;, 
			Shape_t output;, 
			int* kernel_shape;, 
			int* pads;, 
			int* strides;
Scatter=
			Shape_t data;, 
			Shape_t indices;, 
			Shape_t updates;, 
			Shape_t output;, 
			int axis;
Sinh=
			Shape_t input;, 
			Shape_t output;
Cosh=
			Shape_t input;, 
			Shape_t output;
Asinh=
			Shape_t input;, 
			Shape_t output;
Acosh=
			Shape_t input;, 
			Shape_t output;
NonMaxSuppression=
			Shape_t boxes;, 
			Shape_t scores;, 
			Shape_t max_output_boxes_per_class;, 
			Shape_t iou_threshold;, 
			Shape_t score_threshold;, 
			Shape_t selected_indices;, 
			int center_point_box;
Atanh=
			Shape_t input;, 
			Shape_t output;
Sign=
			Shape_t input;, 
			Shape_t output;
Erf=
			Shape_t input;, 
			Shape_t output;
Where=
			Shape_t condition;, 
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t output;
NonZero=
			Shape_t X;, 
			Shape_t Y;
MeanVarianceNormalization=
			Shape_t X;, 
			Shape_t Y;, 
			int* axes;
StringNormalizer=
			Shape_t X;, 
			Shape_t Y;, 
			int case_change_action;, 
			int is_case_sensitive;, 
			int locale;, 
			std::vector<std::string> stopwords;
Mod=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t C;, 
			int fmod;
ThresholdedRelu=
			Shape_t X;, 
			Shape_t Y;, 
			float alpha;
MatMulInteger=
			Shape_t A;, 
			Shape_t B;, 
			Shape_t a_zero_point;, 
			Shape_t b_zero_point;, 
			Shape_t Y;
QLinearMatMul=
			Shape_t a;, 
			Shape_t a_scale;, 
			Shape_t a_zero_point;, 
			Shape_t b;, 
			Shape_t b_scale;, 
			Shape_t b_zero_point;, 
			Shape_t y_scale;, 
			Shape_t y_zero_point;, 
			Shape_t y;
ConvInteger=
			Shape_t x;, 
			Shape_t w;, 
			Shape_t x_zero_point;, 
			Shape_t w_zero_point;, 
			Shape_t y;, 
			int auto_pad;, 
			int* dilations;, 
			int group;, 
			int* kernel_shape;, 
			int* pads;, 
			int* strides;
QLinearConv=
			Shape_t x;, 
			Shape_t x_scale;, 
			Shape_t x_zero_point;, 
			Shape_t w;, 
			Shape_t w_scale;, 
			Shape_t w_zero_point;, 
			Shape_t y_scale;, 
			Shape_t y_zero_point;, 
			Shape_t B;, 
			Shape_t y;, 
			int auto_pad;, 
			int* dilations;, 
			int group;, 
			int* kernel_shape;, 
			int* pads;, 
			int* strides;
QuantizeLinear=
			Shape_t x;, 
			Shape_t y_scale;, 
			Shape_t y_zero_point;, 
			Shape_t y;
DequantizeLinear=
			Shape_t x;, 
			Shape_t x_scale;, 
			Shape_t x_zero_point;, 
			Shape_t y;
IsInf=
			Shape_t X;, 
			Shape_t Y;, 
			int detect_negative;, 
			int detect_positive;
RoiAlign=
			Shape_t X;, 
			Shape_t rois;, 
			Shape_t batch_indices;, 
			Shape_t Y;, 
			int mode;, 
			int output_height;, 
			int output_width;, 
			int sampling_ratio;, 
			float spatial_scale;
ArrayFeatureExtractor=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Z;
Binarizer=
			Shape_t X;, 
			Shape_t Y;, 
			float threshold;
CategoryMapper=
			Shape_t X;, 
			Shape_t Y;, 
			int* cats_int64s;, 
			std::vector<std::string> cats_strings;, 
			int default_int64;, 
			int default_string;
DictVectorizer=
			Shape_t X;, 
			Shape_t Y;, 
			int* int64_vocabulary;, 
			std::vector<std::string> string_vocabulary;
FeatureVectorizer=
			Shape_t X;, 
			Shape_t Y;, 
			int* inputdimensions;
LabelEncoder=
			Shape_t X;, 
			Shape_t Y;, 
			float default_float;, 
			int default_int64;, 
			int default_string;, 
			float* keys_floats;, 
			int* keys_int64s;, 
			std::vector<std::string> keys_strings;, 
			float* values_floats;, 
			int* values_int64s;, 
			std::vector<std::string> values_strings;
LinearClassifier=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Z;, 
			int* classlabels_ints;, 
			std::vector<std::string> classlabels_strings;, 
			float* coefficients;, 
			float* intercepts;, 
			int multi_class;, 
			int post_transform;
LinearRegressor=
			Shape_t X;, 
			Shape_t Y;, 
			float* coefficients;, 
			float* intercepts;, 
			int post_transform;, 
			int targets;
Normalizer=
			Shape_t X;, 
			Shape_t Y;, 
			int norm;
SVMRegressor=
			Shape_t X;, 
			Shape_t Y;, 
			float* coefficients;, 
			float* kernel_params;, 
			int kernel_type;, 
			int n_supports;, 
			int one_class;, 
			int post_transform;, 
			float* rho;, 
			float* support_vectors;
Scaler=
			Shape_t X;, 
			Shape_t Y;, 
			float* offset;, 
			float* scale;
TreeEnsembleClassifier=
			Shape_t X;, 
			Shape_t Y;, 
			Shape_t Z;, 
			float* base_values;, 
			int* class_ids;, 
			int* class_nodeids;, 
			int* class_treeids;, 
			float* class_weights;, 
			int* classlabels_int64s;, 
			std::vector<std::string> classlabels_strings;, 
			int* nodes_falsenodeids;, 
			int* nodes_featureids;, 
			float* nodes_hitrates;, 
			int* nodes_missing_value_tracks_true;, 
			std::vector<std::string> nodes_modes;, 
			int* nodes_nodeids;, 
			int* nodes_treeids;, 
			int* nodes_truenodeids;, 
			float* nodes_values;, 
			int post_transform;
ZipMap=
			Shape_t X;, 
			Shape_t Z;, 
			int* classlabels_int64s;, 
			std::vector<std::string> classlabels_strings;
