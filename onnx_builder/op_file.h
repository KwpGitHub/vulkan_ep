LSTM=activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget
Identity=
Abs=
BatchNormalization=epsilon, momentum
Mean=
Add=
GlobalMaxPool=
Cast=to
AveragePool=kernel_shape, auto_pad, ceil_mode, count_include_pad, pads, strides
And=
LRN=size, alpha, beta, bias
ArgMax=axis, keepdims
Resize=mode
Expand=
Neg=
Mul=
ArgMin=axis, keepdims
CastMap=cast_to, map_form, max_map
Exp=
Div=
ReverseSequence=batch_axis, time_axis
Ceil=
DepthToSpace=blocksize
Clip=max, min
RNN=activation_alpha, activation_beta, activations, clip, direction, hidden_size
Concat=axis
Constant=value
LpPool=kernel_shape, auto_pad, p, pads, strides
Conv=auto_pad, dilations, group, kernel_shape, pads, strides
Not=
Gather=axis
ConvTranspose=auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides
Dropout=ratio
LeakyRelu=alpha
Elu=alpha
GlobalAveragePool=
Gemm=alpha, beta, transA, transB
MaxPool=kernel_shape, auto_pad, ceil_mode, dilations, pads, storage_order, strides
Equal=
Tile=
Flatten=axis
Floor=
GRU=activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset
GlobalLpPool=p
Greater=
HardSigmoid=alpha, beta
Selu=alpha, gamma
Hardmax=axis
If=else_branch, then_branch
Min=
InstanceNormalization=epsilon
Less=
EyeLike=dtype, k
RandomNormal=shape, dtype, mean, scale, seed
Slice=
PRelu=
Log=
LogSoftmax=axis
Loop=body
LpNormalization=axis, p
MatMul=
ReduceL2=axes, keepdims
Max=
MaxRoiPool=pooled_shape, spatial_scale
Or=
Pad=pads, mode, value
RandomUniformLike=dtype, high, low, seed
Reciprocal=
Pow=
RandomNormalLike=dtype, mean, scale, seed
OneHot=axis
RandomUniform=shape, dtype, high, low, seed
ReduceL1=axes, keepdims
ReduceLogSum=axes, keepdims
ReduceLogSumExp=axes, keepdims
ReduceMax=axes, keepdims
OneHotEncoder=cats_int64s, cats_strings, zeros
IsNaN=
ReduceMean=axes, keepdims
ReduceMin=axes, keepdims
TreeEnsembleRegressor=aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
ReduceProd=axes, keepdims
ReduceSum=axes, keepdims
ReduceSumSquare=axes, keepdims
Relu=
Reshape=
Shape=
Sigmoid=
Size=
Softmax=axis
Softplus=
Softsign=
SpaceToDepth=blocksize
TfIdfVectorizer=max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights
Split=axis, split
Imputer=imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
Sqrt=
Squeeze=axes
TopK=axis
Sub=
Sum=
Shrink=bias, lambd
Tanh=
Transpose=perm
Unsqueeze=axes
Upsample=mode
SVMClassifier=classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
Xor=
Acos=
Asin=
Atan=
Cos=
Sin=
Tan=
Multinomial=dtype, sample_size, seed
Scan=body, num_scan_inputs, scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions
Compress=axis
ConstantOfShape=value
MaxUnpool=kernel_shape, pads, strides
Scatter=axis
Sinh=
Cosh=
Asinh=
Acosh=
NonMaxSuppression=center_point_box
Atanh=
Sign=
Erf=
Where=
NonZero=
MeanVarianceNormalization=axes
StringNormalizer=case_change_action, is_case_sensitive, locale, stopwords
Mod=fmod
ThresholdedRelu=alpha
MatMulInteger=
QLinearMatMul=
ConvInteger=auto_pad, dilations, group, kernel_shape, pads, strides
QLinearConv=auto_pad, dilations, group, kernel_shape, pads, strides
QuantizeLinear=
DequantizeLinear=
IsInf=detect_negative, detect_positive
RoiAlign=mode, output_height, output_width, sampling_ratio, spatial_scale
ArrayFeatureExtractor=
Binarizer=threshold
CategoryMapper=cats_int64s, cats_strings, default_int64, default_string
DictVectorizer=int64_vocabulary, string_vocabulary
FeatureVectorizer=inputdimensions
LabelEncoder=default_float, default_int64, default_string, keys_floats, keys_int64s, keys_strings, values_floats, values_int64s, values_strings
LinearClassifier=coefficients, classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
LinearRegressor=coefficients, intercepts, post_transform, targets
Normalizer=norm
SVMRegressor=coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
Scaler=offset, scale
TreeEnsembleClassifier=base_values, class_ids, class_nodeids, class_treeids, class_weights, classlabels_int64s, classlabels_strings, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform
ZipMap=classlabels_int64s, classlabels_strings
