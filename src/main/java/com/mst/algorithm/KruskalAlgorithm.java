package com.mst.algorithm;

import com.mst.model.Edge;
import com.mst.model.Graph;
import com.mst.model.MSTAlgorithmResult;
import java.util.*;

public class KruskalAlgorithm {

    public MSTAlgorithmResult findMST(Graph graph) {
        long startTime = System.nanoTime();
        int operationCount = 0;

        List<Edge> mstEdges = new ArrayList<>();
        int totalCost = 0;

        // Sort edges by weight
        List<Edge> sortedEdges = new ArrayList<>(graph.getEdges());
        sortedEdges.sort(Comparator.comparingInt(Edge::getWeight));
        operationCount += sortedEdges.size() * (int)Math.log(sortedEdges.size());

        UnionFind uf = new UnionFind(graph.getNodes());

        for (Edge edge : sortedEdges) {
            operationCount++;
            String root1 = uf.find(edge.getFrom());
            String root2 = uf.find(edge.getTo());
            operationCount += 2;

            if (!root1.equals(root2)) {
                mstEdges.add(edge);
                totalCost += edge.getWeight();
                uf.union(edge.getFrom(), edge.getTo());
                operationCount += 2;
            }

            if (mstEdges.size() == graph.getNodes().size() - 1) {
                break;
            }
        }

        long endTime = System.nanoTime();
        double executionTimeMs = (endTime - startTime) / 1_000_000.0;

        return new MSTAlgorithmResult(mstEdges, totalCost, operationCount, executionTimeMs);
    }

    // Union-Find (Disjoint Set Union) data structure
    private static class UnionFind {
        private Map<String, String> parent;
        private Map<String, Integer> rank;

        public UnionFind(List<String> nodes) {
            parent = new HashMap<>();
            rank = new HashMap<>();
            for (String node : nodes) {
                parent.put(node, node);
                rank.put(node, 0);
            }
        }

        public String find(String node) {
            if (!parent.get(node).equals(node)) {
                parent.put(node, find(parent.get(node)));
            }
            return parent.get(node);
        }

        public void union(String node1, String node2) {
            String root1 = find(node1);
            String root2 = find(node2);

            if (!root1.equals(root2)) {
                if (rank.get(root1) < rank.get(root2)) {
                    parent.put(root1, root2);
                } else if (rank.get(root1) > rank.get(root2)) {
                    parent.put(root2, root1);
                } else {
                    parent.put(root2, root1);
                    rank.put(root1, rank.get(root1) + 1);
                }
            }
        }
    }
}