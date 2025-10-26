import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
from datetime import datetime
import os

class MSTPerformanceAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.output_dir = self.create_output_structure()
        self.data = self.load_data()
        self.df = self.process_data()

    def create_output_structure(self):
        """Create organized folder structure for results"""
        base_dir = os.path.join(os.path.dirname(__file__), 'results')
        subdirs = ['comprehensive', 'detailed', 'summary', 'comparison']

        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        print(f"📁 Output directory created: {base_dir}")
        return base_dir

    def load_data(self):
        """Load results from JSON file"""
        results_path = os.path.join('..', 'data', 'output', 'results.json')
        print(f"🔍 Looking for results file: {results_path}")
        print(f"📄 File exists: {os.path.exists(results_path)}")

        if not os.path.exists(results_path):
            print(f"❌ ERROR: File not found at {results_path}")
            print("💡 Please run the Java program first: mvn exec:java")
            return {"results": []}

        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            result_count = len(data.get('results', []))
            print(f"✅ Successfully loaded {result_count} graph results")
            return data

    def process_data(self):
        """Process data into pandas DataFrame"""
        records = []

        for result in self.data['results']:
            graph_id = result['graph_id']
            vertices = result['input_stats']['vertices']
            edges = result['input_stats']['edges']

            # Prim's algorithm data
            prim = result['prim']
            records.append({
                'graph_id': graph_id,
                'algorithm': 'Prim',
                'vertices': vertices,
                'edges': edges,
                'total_cost': prim['total_cost'],
                'execution_time_ms': prim['execution_time_ms'],
                'operations_count': prim['operations_count'],
                'mst_edges': len(prim['mst_edges']),
                'density': edges / (vertices * (vertices - 1) / 2) if vertices > 1 else 0
            })

            # Kruskal's algorithm data
            kruskal = result['kruskal']
            records.append({
                'graph_id': graph_id,
                'algorithm': 'Kruskal',
                'vertices': vertices,
                'edges': edges,
                'total_cost': kruskal['total_cost'],
                'execution_time_ms': kruskal['execution_time_ms'],
                'operations_count': kruskal['operations_count'],
                'mst_edges': len(kruskal['mst_edges']),
                'density': edges / (vertices * (vertices - 1) / 2) if vertices > 1 else 0
            })

        df = pd.DataFrame(records)
        print(f"📊 Processed {len(df)} algorithm results ({len(df)//2} graphs)")
        return df

    def theoretical_complexity(self, n, m, algorithm):
        """Calculate theoretical time complexity"""
        if n == 0 or m == 0:
            return 0

        if algorithm == 'Prim':
            # Prim: O((V+E) log V) with binary heap
            return (n + m) * np.log2(n)
        elif algorithm == 'Kruskal':
            # Kruskal: O(E log E) for sorting + O(E α(V)) for Union-Find
            return m * np.log2(m) + m * np.log2(n)
        return 0

    def save_plot(self, fig, filename, subfolder=""):
        """Save plot to organized directory structure"""
        if subfolder:
            path = os.path.join(self.output_dir, subfolder, filename)
        else:
            path = os.path.join(self.output_dir, filename)

        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {path}")
        plt.close(fig)

    def create_performance_overview(self):
        """Create comprehensive performance overview"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MST Algorithms Performance Overview', fontsize=16, fontweight='bold')

        # 1. Execution Time vs Vertices
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            axes[0,0].scatter(algo_data['vertices'], algo_data['execution_time_ms'],
                            label=algorithm, alpha=0.7, s=60)

            if len(algo_data) > 1:
                z = np.polyfit(algo_data['vertices'], algo_data['execution_time_ms'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(algo_data['vertices'].min(), algo_data['vertices'].max(), 100)
                axes[0,0].plot(x_smooth, p(x_smooth), '--', alpha=0.8)

        axes[0,0].set_xlabel('Number of Vertices (V)')
        axes[0,0].set_ylabel('Execution Time (ms)')
        axes[0,0].set_title('Execution Time vs Vertices')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Execution Time vs Edges
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            axes[0,1].scatter(algo_data['edges'], algo_data['execution_time_ms'],
                            label=algorithm, alpha=0.7, s=60)

        axes[0,1].set_xlabel('Number of Edges (E)')
        axes[0,1].set_ylabel('Execution Time (ms)')
        axes[0,1].set_title('Execution Time vs Edges')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Operations Count Comparison
        algorithms = ['Prim', 'Kruskal']
        operations_data = [self.df[self.df['algorithm'] == algo]['operations_count'] for algo in algorithms]
        axes[1,0].boxplot(operations_data, labels=algorithms)
        axes[1,0].set_ylabel('Operations Count')
        axes[1,0].set_title('Operations Count Distribution')
        axes[1,0].grid(True, alpha=0.3)

        # 4. Time Ratio (Kruskal/Prim)
        prim_data = self.df[self.df['algorithm'] == 'Prim'].set_index('graph_id')
        kruskal_data = self.df[self.df['algorithm'] == 'Kruskal'].set_index('graph_id')
        merged_data = prim_data.join(kruskal_data, how='inner', rsuffix='_kruskal')
        merged_data['time_ratio'] = merged_data['execution_time_ms_kruskal'] / merged_data['execution_time_ms']

        axes[1,1].scatter(merged_data['vertices'], merged_data['time_ratio'], alpha=0.7, s=60)
        axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[1,1].set_xlabel('Number of Vertices')
        axes[1,1].set_ylabel('Time Ratio (Kruskal/Prim)')
        axes[1,1].set_title('Performance Ratio: Kruskal vs Prim\n(Ratio > 1: Prim faster)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, 'performance_overview.png', 'comprehensive')

    def create_theoretical_vs_practical(self):
        """Compare theoretical vs practical performance"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Theoretical vs Practical for Prim
        prim_data = self.df[self.df['algorithm'] == 'Prim']
        if len(prim_data) > 0:
            prim_theoretical = [self.theoretical_complexity(row['vertices'], row['edges'], 'Prim')
                              for _, row in prim_data.iterrows()]

            # Normalize
            scale_prim = np.mean(prim_data['execution_time_ms']) / np.mean(prim_theoretical)
            prim_theoretical = [t * scale_prim for t in prim_theoretical]

            axes[0].scatter(prim_theoretical, prim_data['execution_time_ms'],
                          alpha=0.7, s=60, label='Prim', color='blue')

        # Theoretical vs Practical for Kruskal
        kruskal_data = self.df[self.df['algorithm'] == 'Kruskal']
        if len(kruskal_data) > 0:
            kruskal_theoretical = [self.theoretical_complexity(row['vertices'], row['edges'], 'Kruskal')
                                 for _, row in kruskal_data.iterrows()]

            # Normalize
            scale_kruskal = np.mean(kruskal_data['execution_time_ms']) / np.mean(kruskal_theoretical)
            kruskal_theoretical = [t * scale_kruskal for t in kruskal_theoretical]

            axes[0].scatter(kruskal_theoretical, kruskal_data['execution_time_ms'],
                          alpha=0.7, s=60, label='Kruskal', color='orange')

        max_val = max(max(prim_data['execution_time_ms']) if len(prim_data) > 0 else 0,
                     max(kruskal_data['execution_time_ms']) if len(kruskal_data) > 0 else 0)
        axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
        axes[0].set_xlabel('Theoretical Complexity (Normalized)')
        axes[0].set_ylabel('Practical Execution Time (ms)')
        axes[0].set_title('Theoretical vs Practical Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Complexity growth comparison
        if len(prim_data) > 0:
            vertices_range = np.linspace(prim_data['vertices'].min(), prim_data['vertices'].max(), 100)
            avg_edges = prim_data['edges'].mean()

            prim_complexity = [self.theoretical_complexity(v, avg_edges, 'Prim') for v in vertices_range]
            kruskal_complexity = [self.theoretical_complexity(v, avg_edges, 'Kruskal') for v in vertices_range]

            # Normalize for comparison
            prim_complexity = [c / max(prim_complexity) for c in prim_complexity]
            kruskal_complexity = [c / max(kruskal_complexity) for c in kruskal_complexity]

            axes[1].plot(vertices_range, prim_complexity, label='Prim: O((V+E) log V)', linewidth=2)
            axes[1].plot(vertices_range, kruskal_complexity, label='Kruskal: O(E log E)', linewidth=2)
            axes[1].set_xlabel('Number of Vertices (V)')
            axes[1].set_ylabel('Normalized Complexity')
            axes[1].set_title('Theoretical Complexity Growth')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, 'theoretical_vs_practical.png', 'comprehensive')

    def create_detailed_time_analysis(self):
        """Detailed analysis of execution time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Time per operation
        self.df['time_per_operation'] = self.df['execution_time_ms'] / self.df['operations_count']
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            axes[0,0].scatter(algo_data['vertices'], algo_data['time_per_operation'],
                            label=algorithm, alpha=0.7, s=50)
        axes[0,0].set_xlabel('Vertices')
        axes[0,0].set_ylabel('Time per Operation (ms/op)')
        axes[0,0].set_title('Time per Operation vs Graph Size')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Operations per vertex
        self.df['ops_per_vertex'] = self.df['operations_count'] / self.df['vertices']
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            axes[0,1].scatter(algo_data['vertices'], algo_data['ops_per_vertex'],
                            label=algorithm, alpha=0.7, s=50)
        axes[0,1].set_xlabel('Vertices')
        axes[0,1].set_ylabel('Operations per Vertex')
        axes[0,1].set_title('Operations Efficiency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Density impact on performance
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            axes[1,0].scatter(algo_data['density'], algo_data['execution_time_ms'],
                            label=algorithm, alpha=0.7, s=50)
        axes[1,0].set_xlabel('Graph Density')
        axes[1,0].set_ylabel('Execution Time (ms)')
        axes[1,0].set_title('Performance vs Graph Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Cost verification
        prim_data = self.df[self.df['algorithm'] == 'Prim']
        kruskal_data = self.df[self.df['algorithm'] == 'Kruskal']
        if len(prim_data) > 0 and len(kruskal_data) > 0:
            # Match by graph_id
            prim_by_id = prim_data.set_index('graph_id')['total_cost']
            kruskal_by_id = kruskal_data.set_index('graph_id')['total_cost']
            common_ids = prim_by_id.index.intersection(kruskal_by_id.index)

            axes[1,1].scatter(prim_by_id[common_ids], kruskal_by_id[common_ids], alpha=0.7, s=50)
            max_cost = max(prim_by_id[common_ids].max(), kruskal_by_id[common_ids].max())
            axes[1,1].plot([0, max_cost], [0, max_cost], 'r--', alpha=0.7, label='Perfect Match')
            axes[1,1].set_xlabel('Prim MST Cost')
            axes[1,1].set_ylabel('Kruskal MST Cost')
            axes[1,1].set_title('MST Cost Verification\n(Should be identical)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot(fig, 'time_analysis.png', 'detailed')

    def create_algorithm_comparison(self):
        """Direct algorithm comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Performance by graph size category
        size_bins = [0, 50, 200, 500, 1000, 2000, float('inf')]
        size_labels = ['Tiny (<50)', 'Small (50-200)', 'Medium (200-500)',
                      'Large (500-1000)', 'XL (1000-2000)', 'XXL (>2000)']

        comparison_data = self.df.copy()
        comparison_data['size_category'] = pd.cut(comparison_data['vertices'],
                                                bins=size_bins, labels=size_labels)

        pivot_data = comparison_data.pivot_table(
            values='execution_time_ms',
            index='size_category',
            columns='algorithm',
            aggfunc='mean'
        ).dropna()

        pivot_data.plot(kind='bar', ax=axes[0], alpha=0.8)
        axes[0].set_xlabel('Graph Size Category')
        axes[0].set_ylabel('Average Execution Time (ms)')
        axes[0].set_title('Algorithm Performance by Graph Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # 2. Speedup factor
        prim_avg = comparison_data[comparison_data['algorithm'] == 'Prim'].groupby('size_category')['execution_time_ms'].mean()
        kruskal_avg = comparison_data[comparison_data['algorithm'] == 'Kruskal'].groupby('size_category')['execution_time_ms'].mean()

        speedup = (kruskal_avg / prim_avg).dropna()
        axes[1].bar(range(len(speedup)), speedup.values, alpha=0.7, color='green')
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        axes[1].set_xticks(range(len(speedup)))
        axes[1].set_xticklabels(speedup.index, rotation=45)
        axes[1].set_xlabel('Graph Size Category')
        axes[1].set_ylabel('Speedup Factor (Kruskal/Prim)')
        axes[1].set_title('Algorithm Speedup Analysis\n(>1: Prim faster)')
        axes[1].grid(True, alpha=0.3)

        # 3. Memory efficiency (operations count)
        ops_pivot = comparison_data.pivot_table(
            values='operations_count',
            index='size_category',
            columns='algorithm',
            aggfunc='mean'
        ).dropna()

        ops_pivot.plot(kind='bar', ax=axes[2], alpha=0.8)
        axes[2].set_xlabel('Graph Size Category')
        axes[2].set_ylabel('Average Operations Count')
        axes[2].set_title('Operations Count by Graph Size')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self.save_plot(fig, 'algorithm_comparison.png', 'comparison')

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_graphs_analyzed': len(self.df) // 2,
            'performance_summary': {},
            'recommendations': [],
            'statistics': {}
        }

        # Calculate performance metrics
        for algorithm in ['Prim', 'Kruskal']:
            algo_data = self.df[self.df['algorithm'] == algorithm]
            report['performance_summary'][algorithm] = {
                'avg_time_ms': float(algo_data['execution_time_ms'].mean()),
                'std_time_ms': float(algo_data['execution_time_ms'].std()),
                'avg_operations': float(algo_data['operations_count'].mean()),
                'time_per_operation': float((algo_data['execution_time_ms'] / algo_data['operations_count']).mean()),
                'max_vertices': int(algo_data['vertices'].max()),
                'max_edges': int(algo_data['edges'].max())
            }

        # Generate recommendations
        prim_avg = report['performance_summary']['Prim']['avg_time_ms']
        kruskal_avg = report['performance_summary']['Kruskal']['avg_time_ms']

        if prim_avg < kruskal_avg:
            report['recommendations'].append("Prim's algorithm shows better average performance")
            speedup = kruskal_avg / prim_avg
            report['recommendations'].append(f"Prim is {speedup:.2f}x faster than Kruskal on average")
        else:
            report['recommendations'].append("Kruskal's algorithm shows better average performance")
            speedup = prim_avg / kruskal_avg
            report['recommendations'].append(f"Kruskal is {speedup:.2f}x faster than Prim on average")

        # Save report
        report_path = os.path.join(self.output_dir, 'summary', 'performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save statistics as CSV
        stats_path = os.path.join(self.output_dir, 'summary', 'statistics.csv')
        stats_df = self.df.groupby('algorithm').agg({
            'execution_time_ms': ['mean', 'std', 'min', 'max'],
            'operations_count': ['mean', 'std'],
            'vertices': 'mean',
            'edges': 'mean',
            'density': 'mean'
        }).round(2)
        stats_df.to_csv(stats_path)

        # Print summary
        print("=" * 60)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        print(f"Total graphs analyzed: {report['total_graphs_analyzed']}")

        for algo, stats in report['performance_summary'].items():
            print(f"\n{algo}'s Algorithm:")
            print(f"  ⏱️  Average Time: {stats['avg_time_ms']:.2f} ms")
            print(f"  📈 Time Std Dev: {stats['std_time_ms']:.2f} ms")
            print(f"  🔢 Avg Operations: {stats['avg_operations']:,.0f}")
            print(f"  ⚡ Time per Operation: {stats['time_per_operation']:.6f} ms/op")

        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print(f"\n💾 Reports saved in: {self.output_dir}")
        print("=" * 60)

    def run_complete_analysis(self):
        """Run all analysis components"""
        print("🚀 Starting comprehensive MST algorithms analysis...")

        # Create all visualizations
        self.create_performance_overview()
        self.create_theoretical_vs_practical()
        self.create_detailed_time_analysis()
        self.create_algorithm_comparison()

        # Generate reports
        self.generate_performance_report()

        print(f"✅ Analysis complete! Check results in: {self.output_dir}")

def main():
    """Main function to run the analysis"""
    # Use relative path to results.json
    results_file = os.path.join('..', 'data', 'output', 'results.json')

    print("🔍 MST Algorithms Performance Analyzer")
    print("=" * 50)

    analyzer = MSTPerformanceAnalyzer(results_file)

    if len(analyzer.df) == 0:
        print("❌ No data to analyze. Please run Java program first.")
        return

    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()