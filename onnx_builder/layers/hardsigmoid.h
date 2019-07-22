#include <vector>
namespace backend {
    class HardSigmoid {
    public:
        HardSigmoid ();
        ~HardSigmoid();
    private:

		float alpha;
		float beta;
    };
}
