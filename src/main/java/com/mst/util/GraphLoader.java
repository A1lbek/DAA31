package com.mst.util;

import com.mst.model.Graph;
import com.mst.model.Edge;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class GraphLoader {
    private static final ObjectMapper mapper = new ObjectMapper();

    @SuppressWarnings("unchecked")
    public static List<Graph> loadGraphs(String filePath) throws IOException {
        File file = new File(filePath);
        Map<String, Object> jsonMap = mapper.readValue(file, Map.class);
        List<Map<String, Object>> graphsData = (List<Map<String, Object>>) jsonMap.get("graphs");

        return mapper.convertValue(graphsData, new TypeReference<List<Graph>>() {});
    }
}