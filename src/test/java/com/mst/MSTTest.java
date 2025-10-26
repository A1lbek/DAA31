package com.mst;

import com.mst.algorithm.PrimAlgorithm;
import com.mst.algorithm.KruskalAlgorithm;
import com.mst.model.Graph;
import com.mst.model.Edge;
import com.mst.model.GraphResult;
import com.mst.model.InputStats;
import com.mst.model.MSTAlgorithmResult;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class MSTTest {

    @Test
    public void testSmallGraph() {
        // Create a small test graph
        List<String> nodes = Arrays.asList("A", "B", "C", "D");
        List<Edge> edges = Arrays.asList(
                new Edge("A", "B", 1),
                new Edge("A", "C", 4),
                new Edge("B", "C", 2),
                new Edge("B", "D", 5),
                new Edge("C", "D", 3)
        );

        Graph graph = new Graph(1, nodes, edges);
        PrimAlgorithm prim = new PrimAlgorithm();
        KruskalAlgorithm kruskal = new KruskalAlgorithm();

        MSTAlgorithmResult primResult = prim.findMST(graph);
        MSTAlgorithmResult kruskalResult = kruskal.findMST(graph);

        // Test 1: Both algorithms should have same total cost
        assertEquals(primResult.getTotalCost(), kruskalResult.getTotalCost());

        // Test 2: MST should have V-1 edges
        assertEquals(graph.getVertexCount() - 1, primResult.getMstEdges().size());
        assertEquals(graph.getVertexCount() - 1, kruskalResult.getMstEdges().size());

        // Test 3: Execution time should be non-negative
        assertTrue(primResult.getExecutionTimeMs() >= 0);
        assertTrue(kruskalResult.getExecutionTimeMs() >= 0);

        // Test 4: Operation count should be non-negative
        assertTrue(primResult.getOperationsCount() >= 0);
        assertTrue(kruskalResult.getOperationsCount() >= 0);
    }

    @Test
    public void testCorrectMSTCost() {
        // Known graph with known MST cost
        List<String> nodes = Arrays.asList("1", "2", "3", "4");
        List<Edge> edges = Arrays.asList(
                new Edge("1", "2", 10),
                new Edge("1", "3", 6),
                new Edge("1", "4", 5),
                new Edge("2", "3", 15),
                new Edge("3", "4", 4)
        );

        Graph graph = new Graph(2, nodes, edges);
        PrimAlgorithm prim = new PrimAlgorithm();
        KruskalAlgorithm kruskal = new KruskalAlgorithm();

        MSTAlgorithmResult primResult = prim.findMST(graph);
        MSTAlgorithmResult kruskalResult = kruskal.findMST(graph);

        // Known MST cost should be 19 (5 + 4 + 10)
        assertEquals(19, primResult.getTotalCost());
        assertEquals(19, kruskalResult.getTotalCost());
    }

    @Test
    public void testOutputStructure() {
        List<String> nodes = Arrays.asList("A", "B", "C");
        List<Edge> edges = Arrays.asList(
                new Edge("A", "B", 1),
                new Edge("B", "C", 2),
                new Edge("A", "C", 3)
        );

        Graph graph = new Graph(1, nodes, edges);
        PrimAlgorithm prim = new PrimAlgorithm();
        KruskalAlgorithm kruskal = new KruskalAlgorithm();

        MSTAlgorithmResult primResult = prim.findMST(graph);
        MSTAlgorithmResult kruskalResult = kruskal.findMST(graph);

        InputStats inputStats = new InputStats(3, 3);
        GraphResult result = new GraphResult(1, inputStats, primResult, kruskalResult);

        // Test structure
        assertEquals(1, result.getGraphId());
        assertEquals(3, result.getInputStats().getVertices());
        assertEquals(3, result.getInputStats().getEdges());
        assertNotNull(result.getPrim());
        assertNotNull(result.getKruskal());

        // Test algorithm results structure
        assertTrue(primResult.getTotalCost() >= 0);
        assertTrue(primResult.getExecutionTimeMs() >= 0);
        assertTrue(primResult.getOperationsCount() >= 0);
        assertNotNull(primResult.getMstEdges());
    }
}