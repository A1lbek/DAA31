package com.mst;

import com.mst.algorithm.PrimAlgorithm;
import com.mst.algorithm.KruskalAlgorithm;
import com.mst.model.Graph;
import com.mst.model.GraphResult;
import com.mst.model.InputStats;
import com.mst.model.MSTAlgorithmResult;
import com.mst.util.GraphLoader;
import com.mst.util.JSONWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            // Обновленные пути с правильными именами файлов
            String[] inputFiles = {
                    "./data/input/small_dense_graphs.json",
                    "./data/input/medium_dense_graphs.json",
                    "./data/input/large_dense_graphs.json",
                    "./data/input/extra_large_dense_graphs.json"
            };

            List<GraphResult> allResults = new ArrayList<>();

            for (String inputFile : inputFiles) {
                System.out.println("Processing: " + inputFile);

                try {
                    List<Graph> graphs = GraphLoader.loadGraphs(inputFile);
                    PrimAlgorithm prim = new PrimAlgorithm();
                    KruskalAlgorithm kruskal = new KruskalAlgorithm();

                    for (Graph graph : graphs) {
                        System.out.println("  Graph " + graph.getId() +
                                " - Vertices: " + graph.getVertexCount() +
                                ", Edges: " + graph.getEdgeCount());

                        MSTAlgorithmResult primResult = prim.findMST(graph);
                        MSTAlgorithmResult kruskalResult = kruskal.findMST(graph);

                        InputStats inputStats = new InputStats(
                                graph.getVertexCount(),
                                graph.getEdgeCount()
                        );

                        GraphResult graphResult = new GraphResult(
                                graph.getId(),
                                inputStats,
                                primResult,
                                kruskalResult
                        );

                        allResults.add(graphResult);

                        if (primResult.getTotalCost() == kruskalResult.getTotalCost()) {
                            System.out.println("    ✓ Costs match: " + primResult.getTotalCost());
                        } else {
                            System.out.println("    ✗ Costs differ! Prim: " + primResult.getTotalCost() +
                                    ", Kruskal: " + kruskalResult.getTotalCost());
                        }

                        System.out.println("    Prim: " + primResult.getExecutionTimeMs() + "ms, " +
                                primResult.getOperationsCount() + " operations");
                        System.out.println("    Kruskal: " + kruskalResult.getExecutionTimeMs() + "ms, " +
                                kruskalResult.getOperationsCount() + " operations");
                    }
                } catch (IOException e) {
                    System.out.println("  ⚠️  File not found: " + inputFile + " - skipping...");
                    System.out.println("  Make sure the file exists in data/input/ folder");
                }
            }

            if (!allResults.isEmpty()) {
                // Создай папку output если её нет
                new java.io.File("./data/output").mkdirs();

                JSONWriter.writeResults(allResults, "./data/output/results.json");
                JSONWriter.writeCSVSummary(allResults, "./data/output/summary.csv");

                System.out.println("\n✅ Results saved to data/output/");
                System.out.println("📊 Processed " + allResults.size() + " graphs total");
            } else {
                System.out.println("\n❌ No graphs processed. Check input files!");
                System.out.println("📁 Expected files in data/input/:");
                System.out.println("   - small_dense_graphs.json");
                System.out.println("   - medium_dense_graphs.json");
                System.out.println("   - large_dense_graphs.json");
                System.out.println("   - extra_large_dense_graphs.json");
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}