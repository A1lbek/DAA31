package com.mst.algorithm;

import com.mst.model.Edge;
import com.mst.model.Graph;
import com.mst.model.MSTAlgorithmResult;
import java.util.*;

public class PrimAlgorithm {  // ← ДОБАВЬ ЭТУ СТРОЧКУ!

    public MSTAlgorithmResult findMST(Graph graph) {
        long startTime = System.nanoTime();
        int operationCount = 0;

        List<Edge> mstEdges = new ArrayList<>();
        int totalCost = 0;
        Set<String> visited = new HashSet<>();
        PriorityQueue<Edge> pq = new PriorityQueue<>(Comparator.comparingInt(Edge::getWeight));

        // Start from first node
        if (!graph.getNodes().isEmpty()) {
            String startNode = graph.getNodes().get(0);
            visited.add(startNode);
            operationCount++;

            // Add all edges from start node to priority queue
            for (Edge edge : graph.getEdges()) {
                if (edge.getFrom().equals(startNode) || edge.getTo().equals(startNode)) {
                    pq.offer(edge);
                    operationCount++;
                }
            }

            while (!pq.isEmpty() && visited.size() < graph.getNodes().size()) {
                Edge minEdge = pq.poll();
                operationCount++;

                String nextNode = null;
                if (visited.contains(minEdge.getFrom()) && !visited.contains(minEdge.getTo())) {
                    nextNode = minEdge.getTo();
                } else if (visited.contains(minEdge.getTo()) && !visited.contains(minEdge.getFrom())) {
                    nextNode = minEdge.getFrom();
                }

                if (nextNode != null) {
                    visited.add(nextNode);
                    mstEdges.add(minEdge);
                    totalCost += minEdge.getWeight();
                    operationCount += 2;

                    // Add edges from the new node
                    for (Edge edge : graph.getEdges()) {
                        if ((edge.getFrom().equals(nextNode) && !visited.contains(edge.getTo())) ||
                                (edge.getTo().equals(nextNode) && !visited.contains(edge.getFrom()))) {
                            pq.offer(edge);
                            operationCount++;
                        }
                    }
                }
            }
        }

        long endTime = System.nanoTime();
        double executionTimeMs = (endTime - startTime) / 1_000_000.0;

        return new MSTAlgorithmResult(mstEdges, totalCost, operationCount, executionTimeMs);
    }
}  // ← И ЭТУ ЗАКРЫВАЮЩУЮ СКОБКУ!