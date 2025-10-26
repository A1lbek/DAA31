package com.mst.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class GraphResult {
    private int graphId;
    private InputStats inputStats;
    private MSTAlgorithmResult prim;
    private MSTAlgorithmResult kruskal;

    public GraphResult() {}

    public GraphResult(int graphId, InputStats inputStats, MSTAlgorithmResult prim, MSTAlgorithmResult kruskal) {
        this.graphId = graphId;
        this.inputStats = inputStats;
        this.prim = prim;
        this.kruskal = kruskal;
    }

    @JsonProperty("graph_id")
    public int getGraphId() { return graphId; }
    public void setGraphId(int graphId) { this.graphId = graphId; }

    @JsonProperty("input_stats")
    public InputStats getInputStats() { return inputStats; }
    public void setInputStats(InputStats inputStats) { this.inputStats = inputStats; }

    @JsonProperty("prim")
    public MSTAlgorithmResult getPrim() { return prim; }
    public void setPrim(MSTAlgorithmResult prim) { this.prim = prim; }

    @JsonProperty("kruskal")
    public MSTAlgorithmResult getKruskal() { return kruskal; }
    public void setKruskal(MSTAlgorithmResult kruskal) { this.kruskal = kruskal; }
}