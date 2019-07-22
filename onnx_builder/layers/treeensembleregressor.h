#include <vector>
namespace backend {
    class TreeEnsembleRegressor {
    public:
        TreeEnsembleRegressor ();
        ~TreeEnsembleRegressor();
    private:

		std::string aggregate_function;
		std::vector<float> base_values;
		int n_targets;
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
		std::vector<int> target_ids;
		std::vector<int> target_nodeids;
		std::vector<int> target_treeids;
		std::vector<float> target_weights;
    };
}
