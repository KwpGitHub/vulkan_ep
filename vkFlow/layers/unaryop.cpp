#include "unaryop.h"

namespace backend {
	namespace CPU {
		UnaryOP::UnaryOP() {
			one_blob_only = true;
			support_inplace = true;
		}

		template<typename Op>
		static int unary_op_inplace(Mat& a, const Option& opt) {
			Op op;
			int size = a.total();
#pragma omp parallel for num_threads(opt.num_threads)
			for (int i = 0; i < size; ++i)
				a[i] = op(a[i]);

			return 0;
		}


		template<typename T>
		struct unary_op_abs {
			T operator() (const T& x) const { return fabs(x); }
		};

		template<typename T>
		struct unary_op_neg {
			T operator() (const T& x) const { return -x; }
		};

		template<typename T>
		struct unary_op_floor {
			T operator() (const T& x) const { return floor(x); }
		};

		template<typename T>
		struct unary_op_ceil {
			T operator() (const T& x) const { return ceil(x); }
		};

		template<typename T>
		struct unary_op_square {
			T operator() (const T& x) const { return x * x; }
		};

		template<typename T>
		struct unary_op_sqrt {
			T operator() (const T& x) const { return sqrt(x); }
		};

		template<typename T>
		struct unary_op_rsqrt {
			T operator() (const T& x) const { return 1.f / sqrt(x); }
		};

		template<typename T>
		struct unary_op_exp {
			T operator() (const T& x) const { return exp(x); }
		};

		template<typename T>
		struct unary_op_log {
			T operator() (const T& x) const { return log(x); }
		};

		template<typename T>
		struct unary_op_sin {
			T operator() (const T& x) const { return sin(x); }
		};

		template<typename T>
		struct unary_op_cos {
			T operator() (const T& x) const { return cos(x); }
		};

		template<typename T>
		struct unary_op_tan {
			T operator() (const T& x) const { return tan(x); }
		};

		template<typename T>
		struct unary_op_asin {
			T operator() (const T& x) const { return asin(x); }
		};

		template<typename T>
		struct unary_op_acos {
			T operator() (const T& x) const { return acos(x); }
		};

		template<typename T>
		struct unary_op_atan {
			T operator() (const T& x) const { return atan(x); }
		};

		template<typename T>
		struct unary_op_reciprocal {
			T operator() (const T& x) const { return 1.f / x; }
		};


		int UnaryOP::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
		{
			if (op_type == Operation_ABS)
				return unary_op_inplace< unary_op_abs<float> >(bottom_top_blob, opt);

			if (op_type == Operation_NEG)
				return unary_op_inplace< unary_op_neg<float> >(bottom_top_blob, opt);

			if (op_type == Operation_FLOOR)
				return unary_op_inplace< unary_op_floor<float> >(bottom_top_blob, opt);

			if (op_type == Operation_CEIL)
				return unary_op_inplace< unary_op_ceil<float> >(bottom_top_blob, opt);

			if (op_type == Operation_SQUARE)
				return unary_op_inplace< unary_op_square<float> >(bottom_top_blob, opt);

			if (op_type == Operation_SQRT)
				return unary_op_inplace< unary_op_sqrt<float> >(bottom_top_blob, opt);

			if (op_type == Operation_RSQRT)
				return unary_op_inplace< unary_op_rsqrt<float> >(bottom_top_blob, opt);

			if (op_type == Operation_EXP)
				return unary_op_inplace< unary_op_exp<float> >(bottom_top_blob, opt);

			if (op_type == Operation_LOG)
				return unary_op_inplace< unary_op_log<float> >(bottom_top_blob, opt);

			if (op_type == Operation_SIN)
				return unary_op_inplace< unary_op_sin<float> >(bottom_top_blob, opt);

			if (op_type == Operation_COS)
				return unary_op_inplace< unary_op_cos<float> >(bottom_top_blob, opt);

			if (op_type == Operation_TAN)
				return unary_op_inplace< unary_op_tan<float> >(bottom_top_blob, opt);

			if (op_type == Operation_ASIN)
				return unary_op_inplace< unary_op_asin<float> >(bottom_top_blob, opt);

			if (op_type == Operation_ACOS)
				return unary_op_inplace< unary_op_acos<float> >(bottom_top_blob, opt);

			if (op_type == Operation_ATAN)
				return unary_op_inplace< unary_op_atan<float> >(bottom_top_blob, opt);

			if (op_type == Operation_RECIPROCAL)
				return unary_op_inplace< unary_op_reciprocal<float> >(bottom_top_blob, opt);

			return 0;
		}

	}
	
	namespace GPU {

	}
}