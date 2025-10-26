package com.mst.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class MSTAlgorithmResult {
    private List<Edge> mstEdges;
    private int totalCost;
    private int operationsCount;
    private double executionTimeMs;

    public MSTAlgorithmResult() {}

    public MSTAlgorithmResult(List<Edge> mstEdges, int totalCost, int operationsCount, double executionTimeMs) {
        this.mstEdges = mstEdges;
        this.totalCost = totalCost;
        this.operationsCount = operationsCount;
        this.executionTimeMs = executionTimeMs;
    }

    @JsonProperty("mst_edges")
    public List<Edge> getMstEdges() { return mstEdges; }
    public void setMstEdges(List<Edge> mstEdges) { this.mstEdges = mstEdges; }

    @JsonProperty("total_cost")
    public int getTotalCost() { return totalCost; }
    public void setTotalCost(int totalCost) { this.totalCost = totalCost; }

    @JsonProperty("operations_count")
    public int getOperationsCount() { return operationsCount; }
    public void setOperationsCount(int operationsCount) { this.operationsCount = operationsCount; }

    @JsonProperty("execution_time_ms")
    public double getExecutionTimeMs() { return executionTimeMs; }
    public void setExecutionTimeMs(double executionTimeMs) { this.executionTimeMs = executionTimeMs; }
}