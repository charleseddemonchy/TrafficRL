#!/usr/bin/env python
"""
Performance Dashboard for Traffic Light Management with RL.
Visualizes training metrics and agent performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_training_results(results_file):
    """Load training results from a file."""
    try:
        if results_file.endswith('.json'):
            with open(results_file, 'r') as f:
                return json.load(f)
        elif results_file.endswith('.csv'):
            import pandas as pd
            return pd.read_csv(results_file).to_dict(orient='list')
        else:
            print(f"Unsupported file format: {results_file}")
            return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def plot_training_curves(results, output_file=None, show=True):
    """Plot training curves from results."""
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: Episode rewards
        ax1 = fig.add_subplot(gs[0, 0])
        if 'rewards' in results:
            ax1.plot(results['rewards'], alpha=0.6, label='Episode Reward')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.grid(True)
        
        # Plot 2: Average rewards
        ax2 = fig.add_subplot(gs[0, 1])
        if 'avg_rewards' in results:
            ax2.plot(results['avg_rewards'], label='Avg Reward (100 episodes)', color='orange')
            if 'eval_rewards' in results and results['eval_rewards']:
                eval_episodes = np.arange(0, len(results['rewards']), results.get('eval_frequency', 20))[:len(results['eval_rewards'])]
                ax2.scatter(eval_episodes, results['eval_rewards'], color='red', label='Evaluation Reward')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Average Rewards')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Loss values
        ax3 = fig.add_subplot(gs[1, 0])
        if 'loss_values' in results and results['loss_values']:
            ax3.plot(results['loss_values'], label='TD Loss', color='green')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss')
            ax3.grid(True)
        
        # Plot 4: Epsilon decay
        ax4 = fig.add_subplot(gs[1, 1])
        if 'epsilon_values' in results:
            ax4.plot(results['epsilon_values'], label='Epsilon', color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.set_title('Exploration Rate')
            ax4.grid(True)
        
        # Plot 5: Learning rate
        ax5 = fig.add_subplot(gs[2, 0])
        if 'learning_rates' in results:
            ax5.plot(results['learning_rates'], label='Learning Rate', color='brown')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Learning Rate')
            ax5.set_title('Optimizer Learning Rate')
            ax5.grid(True)
        
        # Plot 6: Performance metrics
        ax6 = fig.add_subplot(gs[2, 1])
        if 'waiting_times' in results and 'throughput' in results:
            ax6.plot(results['waiting_times'], label='Avg Waiting Time', color='red', alpha=0.7)
            ax6_2 = ax6.twinx()
            ax6_2.plot(results['throughput'], label='Throughput', color='blue', alpha=0.7)
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Waiting Time', color='red')
            ax6_2.set_ylabel('Throughput', color='blue')
            ax6.set_title('Traffic Performance Metrics')
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_2.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax6.grid(True)
        
        # Add overall title
        fig.suptitle('Traffic Light RL Agent Training Performance', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show the figure
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=150)
            print(f"Dashboard saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error plotting training curves: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def plot_policy_heatmap(analysis_file, output_file=None, show=True):
    """Plot policy heatmap from analysis results."""
    try:
        # Load analysis results
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Check if decision boundaries data exists
        if 'decision_boundaries' not in analysis:
            print("No decision boundary data found in analysis file")
            return False
        
        # Create figure with subplots for each light state
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Custom colormap for the two actions
        cmap = plt.cm.get_cmap('RdYlGn')
        
        for i, light_state in enumerate([0, 1]):
            # Extract data for this light state
            key = f"light_{light_state}"
            if key not in analysis['decision_boundaries']:
                print(f"No data for light state {light_state}")
                continue
                
            data = analysis['decision_boundaries'][key]
            
            # Extract NS and EW densities and actions
            ns_values = [point['ns_density'] for point in data]
            ew_values = [point['ew_density'] for point in data]
            actions = [point['action'] for point in data]
            
            # Determine grid dimensions
            grid_size = int(np.sqrt(len(data)))
            
            # Reshape data into a grid
            ns_grid = np.reshape(np.unique(ns_values), (grid_size, 1))
            ew_grid = np.reshape(np.unique(ew_values), (1, grid_size))
            action_grid = np.zeros((grid_size, grid_size))
            
            # Fill action grid
            for idx, point in enumerate(data):
                row = int(idx / grid_size)
                col = idx % grid_size
                action_grid[row, col] = point['action']
            
            # Plot heatmap
            im = axes[i].imshow(
                action_grid, 
                origin='lower', 
                cmap=cmap,
                extent=[0, 1, 0, 1],
                aspect='auto',
                vmin=0,
                vmax=1
            )
            
            # Add labels and title
            axes[i].set_xlabel('East-West Density')
            axes[i].set_ylabel('North-South Density')
            axes[i].set_title(f'Current Light State: {"NS Green" if light_state == 0 else "EW Green"}')
            
            # Add grid
            axes[i].grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Add contour to show decision boundary
            cs = axes[i].contour(action_grid, levels=[0.5], colors='black', 
                              extent=[0, 1, 0, 1], linewidths=2)
            axes[i].clabel(cs, inline=1, fontsize=10)
        
        # Add colorbar with labels
        cbar = fig.colorbar(im, ax=axes, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['NS Green', 'Threshold', 'EW Green'])
        
        # Add overall title
        fig.suptitle('Traffic Light Decision Policy by Traffic Density', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the figure
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=150)
            print(f"Policy heatmap saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error plotting policy heatmap: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def plot_performance_comparison(analysis_file, output_file=None, show=True):
    """Plot performance comparison across traffic patterns."""
    try:
        # Load analysis results
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Check if performance metrics data exists
        if 'performance_metrics' not in analysis:
            print("No performance metrics found in analysis file")
            return False
        
        metrics = analysis['performance_metrics']
        traffic_patterns = list(metrics.keys())
        
        if not traffic_patterns:
            print("No traffic patterns found in performance metrics")
            return False
        
        # Metrics to plot
        metric_keys = ['avg_reward', 'avg_waiting_time', 'avg_cars_passed', 'avg_density']
        metric_titles = ['Average Reward', 'Average Waiting Time', 'Average Cars Passed', 'Average Traffic Density']
        metric_colors = ['green', 'red', 'blue', 'purple']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (key, title, color) in enumerate(zip(metric_keys, metric_titles, metric_colors)):
            values = [metrics[pattern].get(key, 0) for pattern in traffic_patterns]
            bars = axes[i].bar(traffic_patterns, values, color=color, alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f"{height:.2f}", ha='center', va='bottom')
            
            axes[i].set_title(title)
            axes[i].set_ylabel(title)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add overall title
        fig.suptitle('Agent Performance Across Different Traffic Patterns', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the figure
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=150)
            print(f"Performance comparison saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error plotting performance comparison: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Performance Dashboard for Traffic Light RL')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to training results file (CSV or JSON)')
    parser.add_argument('--analysis', type=str, default=None,
                        help='Path to agent analysis file (JSON)')
    parser.add_argument('--output-dir', type=str, default='dashboards',
                        help='Directory to save dashboard plots')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training results
    results = load_training_results(args.results)
    if results:
        output_file = os.path.join(args.output_dir, 'training_dashboard.png')
        plot_training_curves(results, output_file, show=not args.no_show)
    
    # Process agent analysis
    if args.analysis and os.path.exists(args.analysis):
        # Plot policy heatmap
        policy_file = os.path.join(args.output_dir, 'policy_heatmap.png')
        plot_policy_heatmap(args.analysis, policy_file, show=not args.no_show)
        
        # Plot performance comparison
        perf_file = os.path.join(args.output_dir, 'performance_comparison.png')
        plot_performance_comparison(args.analysis, perf_file, show=not args.no_show)
    
if __name__ == "__main__":
    main()