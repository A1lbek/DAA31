package com.mst.util;

import com.mst.model.GraphResult;
import com.mst.model.InputStats;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class JSONWriter {
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void writeResults(List<GraphResult> results, String filePath) throws IOException {
        Map<String, Object> output = new HashMap<>();
        output.put("results", results);

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(filePath), output);
    }

    public static void writeCSVSummary(List<GraphResult> results, String filePath) throws IOException {
        StringBuilder csv = new StringBuilder();
        csv.append("GraphID,Vertices,Edges,PrimCost,PrimTimeMs,PrimOperations,KruskalCost,KruskalTimeMs,KruskalOperations\n");

        for (GraphResult result : results) {
            csv.append(result.getGraphId()).append(",")
                    .append(result.getInputStats().getVertices()).append(",")
                    .append(result.getInputStats().getEdges()).append(",")
                    .append(result.getPrim().getTotalCost()).append(",")
                    .append(result.getPrim().getExecutionTimeMs()).append(",")
                    .append(result.getPrim().getOperationsCount()).append(",")
                    .append(result.getKruskal().getTotalCost()).append(",")
                    .append(result.getKruskal().getExecutionTimeMs()).append(",")
                    .append(result.getKruskal().getOperationsCount()).append("\n");
        }

        java.nio.file.Files.write(
                java.nio.file.Paths.get(filePath),
                csv.toString().getBytes()
        );
    }
}