package com.mst.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class Graph {
    private int id;
    private List<String> nodes;
    private List<Edge> edges;

    public Graph() {}

    public Graph(int id, List<String> nodes, List<Edge> edges) {
        this.id = id;
        this.nodes = nodes;
        this.edges = edges;
    }

    @JsonProperty("id")
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    @JsonProperty("nodes")
    public List<String> getNodes() { return nodes; }
    public void setNodes(List<String> nodes) { this.nodes = nodes; }

    @JsonProperty("edges")
    public List<Edge> getEdges() { return edges; }
    public void setEdges(List<Edge> edges) { this.edges = edges; }

    public int getVertexCount() {
        return nodes != null ? nodes.size() : 0;
    }

    public int getEdgeCount() {
        return edges != null ? edges.size() : 0;
    }
}