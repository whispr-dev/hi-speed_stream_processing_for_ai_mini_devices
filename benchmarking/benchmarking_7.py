import json
import pandas as pd
from typing import Dict, List, Any
import numpy as np

class BenchmarkResultParser:
    def __init__(self, results_file: str):
        """
        Initialize parser with benchmark results file
        
        :param results_file: Path to JSON results file
        """
        self.raw_results = self._load_results(results_file)
        self.processed_data = {}

    def _load_results(self, results_file: str) -> List[Dict[str, Any]]:
        """
        Load JSON results from file
        
        :param results_file: Path to results file
        :return: List of benchmark result dictionaries
        """
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading results: {e}")
            return []

    def normalize_results(self) -> Dict[str, Any]:
        """
        Normalize and aggregate benchmark results
        
        :return: Structured, normalized benchmark data
        """
        normalized = {
            'hardware_configs': [],
            'performance_metrics': {
                'latency': [],
                'cpu_utilization': [],
                'memory_bandwidth': [],
                'power_consumption': []
            },
            'signal_quality': {
                'pesq_scores': [],
                'stoi_scores': [],
                'spectral_distortion': []
            }
        }

        for result in self.raw_results:
            # Collect hardware configurations
            normalized['hardware_configs'].append({
                'device_type': result.get('hardware_config', {}).get('device_type', 'Unknown'),
                'cpu_model': result.get('hardware_config', {}).get('cpu_model', 'Unknown')
            })

            # Aggregate performance metrics
            perf = result.get('preprocessing_performance', {})
            normalized['performance_metrics']['latency'].append(perf.get('total_latency', 0))
            normalized['performance_metrics']['cpu_utilization'].append(perf.get('cpu_utilization', 0))
            normalized['performance_metrics']['memory_bandwidth'].append(perf.get('memory_bandwidth', 0))
            normalized['performance_metrics']['power_consumption'].append(perf.get('power_consumption', 0))

            # Aggregate signal quality metrics
            quality = result.get('signal_quality_metrics', {})
            normalized['signal_quality']['pesq_scores'].append(quality.get('pesq_score', 0))
            normalized['signal_quality']['stoi_scores'].append(quality.get('stoi_score', 0))
            normalized['signal_quality']['spectral_distortion'].append(quality.get('spectral_distortion', 0))

        # Calculate statistical summaries
        for category in normalized['performance_metrics']:
            normalized['performance_metrics'][category + '_stats'] = {
                'mean': np.mean(normalized['performance_metrics'][category]),
                'median': np.median(normalized['performance_metrics'][category]),
                'std_dev': np.std(normalized['performance_metrics'][category])
            }

        for category in normalized['signal_quality']:
            if category.endswith('scores') or category == 'spectral_distortion':
                normalized['signal_quality'][category + '_stats'] = {
                    'mean': np.mean(normalized['signal_quality'][category]),
                    'median': np.median(normalized['signal_quality'][category]),
                    'std_dev': np.std(normalized['signal_quality'][category])
                }

        self.processed_data = normalized
        return normalized

    def generate_comparative_analysis(self) -> Dict[str, Any]:
        """
        Generate comparative analysis between different benchmark runs
        
        :return: Comparative performance insights
        """
        if not self.processed_data:
            self.normalize_results()

        comparative_analysis = {
            'best_performing_config': {},
            'performance_improvements': {}
        }

        # Find best performing hardware configuration
        performance_scores = []
        for i, config in enumerate(self.processed_data['hardware_configs']):
            # Simple performance scoring (lower is better for latency, higher is better for quality)
            score = (
                self.processed_data['performance_metrics']['latency'][i] * -1 +
                self.processed_data['signal_quality']['pesq_scores'][i]
            )
            performance_scores.append((score, config))

        # Sort and get top performer
        if performance_scores:
            best_score, best_config = max(performance_scores, key=lambda x: x[0])
            comparative_analysis['best_performing_config'] = best_config

        return comparative_analysis

# Example usage
def main():
    parser = BenchmarkResultParser('benchmark_results.json')
    normalized_data = parser.normalize_results()
    comparative_analysis = parser.generate_comparative_analysis()
    
    print(json.dumps(normalized_data, indent=2))
    print("\nComparative Analysis:")
    print(json.dumps(comparative_analysis, indent=2))

if __name__ == '__main__':
    main()