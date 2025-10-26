package com.mst.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class InputStats {
    private int vertices;
    private int edges;

    public InputStats() {}

    public InputStats(int vertices, int edges) {
        this.vertices = vertices;
        this.edges = edges;
    }

    @JsonProperty("vertices")
    public int getVertices() { return vertices; }
    public void setVertices(int vertices) { this.vertices = vertices; }

    @JsonProperty("edges")
    public int getEdges() { return edges; }
    public void setEdges(int edges) { this.edges = edges; }
}