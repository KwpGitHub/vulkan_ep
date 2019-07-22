#include <vector>
namespace backend {
    class TreeEnsembleClassifier {
    public:
        TreeEnsembleClassifier ();
        ~TreeEnsembleClassifier();
    private:

		std::vector<float> base_values;
		std::vector<int> class_ids;
		std::vector<int> class_nodeids;
		std::vector<int> class_treeids;
		std::vector<float> class_weights;
		std::vector<int> classlabels_int64s;
		std::vector<string> classlabels_strings;
		std::vector<int> nodes_falsenodeids;
		std::vector<int> nodes_featureids;
		std::vector<float> nodes_hitrates;
		std::vector<int> nodes_missing_value_tracks_true;
		std::vector<string> nodes_modes;
		std::vector<int> nodes_nodeids;
		std::vector<int> nodes_treeids;
		std::vector<int> nodes_truenodeids;
		std::vector<float> nodes_values;
		std::string post_transform;
    };
}
