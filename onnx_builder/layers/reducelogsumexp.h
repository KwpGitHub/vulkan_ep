#include <vector>
namespace backend {
    class ReduceLogSumExp {
    public:
        ReduceLogSumExp ();
        ~ReduceLogSumExp();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
