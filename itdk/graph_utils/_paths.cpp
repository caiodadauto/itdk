#include <iostream>
#include <numeric>
#include <vector>
#include "_paths.h"


void dfs_to_target(int source, int target, const std::vector<std::vector<float>>& WU, std::vector<short>& visited, std::vector<std::vector<float>>& paths, std::vector<float>& path)
{
    int num_of_nodes = WU.size();
    float weight;

    path.push_back(source);
    visited[source] = 1;
    if(source == target) {
        paths.push_back(path);
        return;
    }
    for(int node = 0; node < num_of_nodes; node++) {
        weight = 0.0;
        if(visited[node] == 0) {
            if(source > node) {
                weight = WU[node][source - node];
            }
            else if(source < node) {
                weight = WU[source][node - source];
            }
            if(weight > 0) {
                path[0] += weight;
                dfs_to_target(node, target, WU, visited, paths, path);
                visited[node] = 0;
                path[0] -= weight;
                path.pop_back();
            }
        }
    }
    return;
}

std::vector<std::vector<float>> paths_dfs(int source, int target, const std::vector<std::vector<float>>& WU)
{
    int num_of_nodes = WU.size();
    std::vector<float> path;
    std::vector<std::vector<float>> paths;
    std::vector<short> visited (num_of_nodes, 0);

    path.push_back(0.0);
    dfs_to_target(source, target, WU, visited, paths, path);
    return paths;
}

std::vector<std::vector<float>> cpp_all_paths(const std::vector<std::vector<float>>& WU, const std::vector<int>& targets, const int n_threads)
{
#pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert( omp_out.end(), omp_in.begin(), omp_in.end()))
    int target = targets[0];
    int num_of_nodes = WU.size();
    std::vector<std::vector<float>> paths;

#pragma omp parallel for reduction(merge: paths) num_threads(n_threads)
    for(int source = 0; source < num_of_nodes; source++) {
        std::vector<std::vector<float>> paths_from_source;
        if(source != target) {
            paths_from_source = paths_dfs(source, target, WU);
            paths.insert(paths.end(), paths_from_source.begin(), paths_from_source.end());
        }
    }

    return paths;
}
