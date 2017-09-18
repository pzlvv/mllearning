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
    int tag;
    int feature_index;
    map<int, DTreeNode*> children;
};


typedef pair<vector<int>, int> DataItem;
typedef list<int> FeatureSet;
typedef list<DataItem> DataSet;

/*Impirical Entropy*/
float H(const DataSet& dataset,
	const DataSet::const_iterator& begin,
	const DataSet::const_iterator& end)
{
    map<int, int> stats;
    for (auto itr=begin; itr!=end; ++itr)
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
	 const DataSet::const_iterator& begin,
	 const DataSet::const_iterator& end,
	 int feature_index)
{
    float impirical_entropy = H(dataset, begin, end);
    map<int, int> stats_x;
    map<int, map<int, int>> stats_y;
    map<int, int> total_y;
    for (auto itr=begin; itr!=end; ++itr)
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

list<DataSet::iterator> multi_partition(DataSet& dataset,
		     const DataSet::iterator& begin,
		     const DataSet::iterator& end,
		     int feature_index)
{
    map<int, DataSet::iterator> block_tail;
    list<int> order;

    auto itr = begin;

    while (itr != end)
    {
	auto next = itr;
	++next;
	int k = (itr->first)[feature_index];
	auto p = block_tail.find(k); 
	if (p == block_tail.end())
	{
	    block_tail[k] = itr;
	    order.push_back(k);
	}
	else
	{
	    auto t = p->second;
	    ++t;
	    dataset.splice(t, dataset, itr);
	    ++(p->second);
	}
	itr = next;
    }

    list<DataSet::iterator> result;
    for (auto itr=order.begin(); itr!=order.end(); ++itr)
    {
	result.push_back(++block_tail[*itr]);
    }
    return result;
}

int most_frequent_category(const DataSet& dataset,
			   const DataSet::iterator& begin,
			   const DataSet::iterator& end)
{
    map<int, int> stats;
    for (auto itr=begin; itr!=end; ++itr)
    {
	++stats[itr->second];
    }
    return max_element(stats.cbegin(), stats.cend(),
		[](const pair<int, int>& lhs, const pair<int, int>& rhs)
		{
		    return lhs.second < rhs.second;
		})->first;
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

    int predict(DTreeNode* root, const vector<int>& x)
    {
	DTreeNode* cur = root;
	while (!cur->children.empty())
	{
	    cur = cur->children[x[cur->feature_index]];
	}
	return cur->tag;
    }

    DTreeNode* GenerateTree(DataSet& dataset,
			    const DataSet::iterator& begin,
			    const DataSet::iterator& end,
			    set<int>& features)
    {
	if (features.empty())
	{
	    int c = most_frequent_category(dataset, begin, end);
	    DTreeNode* node = new DTreeNode;
	    node->tag = c;
	    node->feature_index = -1;
	    return node;
	}

	int target = dataset.cbegin()->second;
	bool all_same_flag = true;
	for (auto itr=dataset.cbegin(); itr!=dataset.cend(); ++itr)
	{
	    if (itr->second != target)
	    {
		all_same_flag = false;
		break;
	    }
	}

	if (all_same_flag)
	{
	    DTreeNode* node = new DTreeNode;
	    node->tag = target;
	    node->feature_index = -1;
	    return node;
	}

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
	 * */
	int ig_max_feature = *ig_max_it;
	list<DataSet::iterator> partition_list = multi_partition(dataset, begin, end, ig_max_feature);
	auto l = begin;
	features.erase(ig_max_it);
	DTreeNode* node = new DTreeNode;
	node->feature_index = ig_max_feature;
	for (auto itr=partition_list.begin(); itr!=partition_list.end(); ++itr)
	{
	    auto x = GenerateTree(dataset, l, *itr, features);
	    //int c = most_frequent_category(dataset, r, *itr);
	    node->children[l->first[ig_max_feature]] = x;
	    l = *itr;
	}
	return node;

    }

};

void print_tree(DTreeNode* node)
{
    if (node->children.empty())
    {
	int tag = node->tag;
	cout << "tag: " << tag << endl;;
    }
    else
    {
	cout << "max feature: " << node->feature_index << endl;
	for (auto itr=node->children.begin(); itr!=node->children.end(); ++itr)
	{
	    cout << "feature value: " << itr->first << endl;
	    print_tree(itr->second);
	}
    }
}

int main()
{
    DataSet dataset;
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
    auto node = dtree.GenerateTree(dataset, dataset.begin(), dataset.end(), features);
    print_tree(node);
    for(auto itr=dataset.begin(); itr!=dataset.end(); ++itr)
    {
	cout << dtree.predict(node, itr->first) << "," << itr->second << endl;
    }
    cout << dtree.predict(node, {0, 0, 0, 0})<<endl;
    return 0;
}
