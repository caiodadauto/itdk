#ifndef _PATH_H
#define _PATH_H

#include<vector>

void dfs_to_target(int source, int target, const std::vector<std::vector<float> >& WU, std::vector<short>& visited, std::vector<std::vector<float> >& paths, std::vector<float>& path);
std::vector<std::vector<float> > paths_dfs(int source, int target, const std::vector<std::vector<float> >& WU);
std::vector<std::vector<float> > cpp_all_paths(const std::vector<std::vector<float> >& WU, const std::vector<int>& targets, const int n_threads);

#endif
