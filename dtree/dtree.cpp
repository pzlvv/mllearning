#include <vector>
#include <utility>
#include <list>
#include <map>
#include <iostream>
#include <cmath>
#include <set>
#include <algorithm>

using namespace std;

class DTreeNode
{
    public:
    private:
        int tag;
        int feature_index;
        vector<DTreeNode*> children;
};


typedef pair<vector<int>, int> DataItem;
typedef list<int> FeatureSet;
typedef vector<DataItem> DataSet;

/*Impirical Entropy*/
float H(const DataSet& dataset,
        const DataSet::const_iterator& it_begin,
        const DataSet::const_iterator& it_end)
{
    map<int, int> stats;
    for (auto itr=it_begin; itr!=it_end; ++itr)
    {
        ++stats[itr->second];
    }
    float sum = 0;
    for (auto itr=stats.begin(); itr!=stats.end(); ++itr)
    {
        float p = itr->second / float(dataset.size());
        sum -= p*log2(p);
    }
    return sum;
}

float IG(const DataSet& dataset,
        const DataSet::const_iterator& it_begin,
        const DataSet::const_iterator& it_end,
        int feature_index)
{
    float impirical_entropy = H(dataset, it_begin, it_end);
    map<int, int> stats_x;
    map<int, map<int, int>> stats_y;
    map<int, int> total_y;
    for (auto itr=it_begin; itr!=it_end; ++itr)
    {
        ++stats_x[(itr->first)[feature_index]];
        ++stats_y[(itr->first)[feature_index]][itr->second];
        ++total_y[(itr->first)[feature_index]];
    }
    float impirical_conditional_entropy = 0;
    for (auto itr=stats_x.begin(); itr!=stats_x.end(); ++itr)
    {
        float p = itr->second / float(dataset.size());
        float sum = 0;
        for (auto itr_y=stats_y[itr->first].begin(); itr_y!=stats_y[itr->first].end(); ++itr_y)
        {
            float p = itr_y->second / float(total_y[itr->first]);
            sum -= p*log2(p);
        }
        impirical_conditional_entropy += p*sum;
    }
    return impirical_entropy - impirical_conditional_entropy;
}

class DTree
{
    public:
        DTree()
        {
        }

        ~DTree()
        {
        }

        DTreeNode* GenerateNode(DataSet& dataset,
                const DataSet::iterator& begin,
                const DataSet::iterator& end,
                set<int>& features)
        {
            /**
             * More Efficient
             * */
            float ig_max = -1;
            set<int>::iterator ig_max_it;
            for (auto itr=features.begin(); itr!=features.end(); ++itr)
            {
                float ig = IG(dataset, begin, end, *itr);
                if (ig > ig_max)
                {
                    ig_max = ig, ig_max_it = itr;
                }
            }

            /**
             * More elegent
            auto ig_max_it = max_element(features.begin(), features.end(),
                    [&](int lhs, int rhs){return IG(dataset, begin, end, lhs) < IG(dataset,begin, end, rhs);});
            */
            
        }

};

int main()
{
    vector<DataItem> dataset;
    dataset.push_back({{0, 0, 0, 0}, 0});
    dataset.push_back({{0, 0, 0, 1}, 0});
    dataset.push_back({{0, 1, 0, 1}, 1});
    dataset.push_back({{0, 1, 1, 0}, 1});
    dataset.push_back({{0, 0, 0, 0}, 0});
    dataset.push_back({{1, 0, 0, 0}, 0});
    dataset.push_back({{1, 0, 0, 1}, 0});
    dataset.push_back({{1, 1, 1, 1}, 1});
    dataset.push_back({{1, 0, 1, 2}, 1});
    dataset.push_back({{1, 0, 1, 2}, 1});
    dataset.push_back({{2, 0, 1, 2}, 1});
    dataset.push_back({{2, 0, 1, 1}, 1});
    dataset.push_back({{2, 1, 0, 1}, 1});
    dataset.push_back({{2, 1, 0, 2}, 1});
    dataset.push_back({{2, 0, 0, 0}, 0});
    //cout << H(dataset) << endl;
    //cout << IG(dataset, dataset.cbegin(), dataset.cend(), 0) << endl;
    DTree dtree;
    set<int> features = {0, 1, 2, 3};
    dtree.GenerateNode(dataset, dataset.begin(), dataset.end(), features);
    return 0;
}
