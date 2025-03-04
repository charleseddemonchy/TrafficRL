"""
Analysis Utilities
================
Comprehensive tools for analyzing agent behavior and performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime

logger = logging.getLogger("TrafficRL.Utils.Analysis")

def analyze_training_metrics(metrics, save_dir="results"):
    """
    Analyze and visualize training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract metrics
        rewards = metrics.get("rewards", [])
        avg_rewards = metrics.get("avg_rewards", [])
        eval_rewards = metrics.get("eval_rewards", [])
        losses = metrics.get("loss_values", [])
        epsilons = metrics.get("epsilon_values", [])
        learning_rates = metrics.get("learning_rates", [])
        waiting_times = metrics.get("waiting_times", [])
        throughputs = metrics.get("throughput", [])
        
        # Calculate summary statistics
        summary = {
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)) if rewards else None,
                "max": float(np.max(rewards)) if rewards else None,
                "final_avg": float(np.mean(rewards[-100:])) if len(rewards) > 100 else float(np.mean(rewards))
            },
            "waiting_time": {
                "mean": float(np.mean(waiting_times)) if waiting_times else None,
                "min": float(np.min(waiting_times)) if waiting_times else None,
                "max": float(np.max(waiting_times)) if waiting_times else None
            },
            "throughput": {
                "mean": float(np.mean(throughputs)) if throughputs else None,
                "min": float(np.min(throughputs)) if throughputs else None,
                "max": float(np.max(throughputs)) if throughputs else None
            },
            "training_time": metrics.get("training_time", 0)
        }
        
        # Plot rewards
        if rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, alpha=0.6, label='Episode Reward')
            plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
            if eval_rewards:
                # Plot evaluation rewards at their corresponding episodes
                eval_episodes = [i * metrics.get("eval_frequency", 20) for i in range(len(eval_rewards))]
                plt.plot(eval_episodes, eval_rewards, 'ro-', label='Evaluation Reward')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "reward_plot.png"))
            plt.close()
        
        # Plot losses
        if losses:
            plt.figure(figsize=(12, 6))
            plt.plot(losses)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "loss_plot.png"))
            plt.close()
        
        # Plot epsilon decay
        if epsilons:
            plt.figure(figsize=(12, 6))
            plt.plot(epsilons)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate (Epsilon) Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "epsilon_plot.png"))
            plt.close()
        
        # Plot learning rate
        if learning_rates:
            plt.figure(figsize=(12, 6))
            plt.plot(learning_rates)
            plt.xlabel('Episode')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "learning_rate_plot.png"))
            plt.close()
        
        # Plot waiting times and throughput
        if waiting_times and throughputs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(waiting_times)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Waiting Time')
            ax1.set_title('Average Waiting Time Per Episode')
            ax1.grid(True)
            
            ax2.plot(throughputs)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Throughput')
            ax2.set_title('Average Throughput Per Episode')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "performance_metrics.png"))
            plt.close()
        
        # Save summary statistics
        with open(os.path.join(save_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Training analysis completed and saved to {save_dir}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error analyzing training metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def comparative_analysis(results, save_dir="results"):
    """
    Analyze and visualize comparative benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract agent types and traffic patterns
        agent_types = set()
        traffic_patterns = set()
        
        for key in results.keys():
            if key == "summary":
                continue
                
            parts = key.split('_')
            if len(parts) >= 2:
                agent_type = parts[0]
                pattern = '_'.join(parts[1:])
                
                agent_types.add(agent_type)
                traffic_patterns.add(pattern)
        
        agent_types = sorted(list(agent_types))
        traffic_patterns = sorted(list(traffic_patterns))
        
        # Prepare data for bar plots
        rewards_data = {}
        waiting_data = {}
        throughput_data = {}
        
        for pattern in traffic_patterns:
            rewards_data[pattern] = []
            waiting_data[pattern] = []
            throughput_data[pattern] = []
            
            for agent in agent_types:
                key = f"{agent}_{pattern}"
                if key in results:
                    result = results[key]
                    rewards_data[pattern].append(result.get("avg_reward", 0))
                    waiting_data[pattern].append(result.get("avg_waiting_time", 0))
                    throughput_data[pattern].append(result.get("avg_throughput", 0))
                else:
                    rewards_data[pattern].append(0)
                    waiting_data[pattern].append(0)
                    throughput_data[pattern].append(0)
        
        # Create comparative bar plots
        bar_width = 0.35
        index = np.arange(len(agent_types))
        
        # Plot reward comparison
        plt.figure(figsize=(12, 6))
        for i, pattern in enumerate(traffic_patterns):
            offset = (i - len(traffic_patterns)/2 + 0.5) * bar_width
            plt.bar(index + offset, rewards_data[pattern], bar_width, label=pattern)
        
        plt.xlabel('Agent Type')
        plt.ylabel('Average Reward')
        plt.title('Reward Comparison by Agent and Traffic Pattern')
        plt.xticks(index, agent_types)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "reward_comparison.png"))
        plt.close()
        
        # Plot waiting time comparison
        plt.figure(figsize=(12, 6))
        for i, pattern in enumerate(traffic_patterns):
            offset = (i - len(traffic_patterns)/2 + 0.5) * bar_width
            plt.bar(index + offset, waiting_data[pattern], bar_width, label=pattern)
        
        plt.xlabel('Agent Type')
        plt.ylabel('Average Waiting Time')
        plt.title('Waiting Time Comparison by Agent and Traffic Pattern')
        plt.xticks(index, agent_types)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "waiting_time_comparison.png"))
        plt.close()
        
        # Plot throughput comparison
        plt.figure(figsize=(12, 6))
        for i, pattern in enumerate(traffic_patterns):
            offset = (i - len(traffic_patterns)/2 + 0.5) * bar_width
            plt.bar(index + offset, throughput_data[pattern], bar_width, label=pattern)
        
        plt.xlabel('Agent Type')
        plt.ylabel('Average Throughput')
        plt.title('Throughput Comparison by Agent and Traffic Pattern')
        plt.xticks(index, agent_types)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, "throughput_comparison.png"))
        plt.close()
        
        # Calculate improvement percentages
        improvements = {}
        for pattern in traffic_patterns:
            if len(agent_types) >= 2:  # Need at least two agents to compare
                baseline_idx = agent_types.index("FixedTiming") if "FixedTiming" in agent_types else 0
                trained_idx = agent_types.index("TrainedDQN") if "TrainedDQN" in agent_types else 1
                
                baseline_reward = rewards_data[pattern][baseline_idx]
                trained_reward = rewards_data[pattern][trained_idx]
                
                if baseline_reward != 0:
                    reward_improvement = (trained_reward - baseline_reward) / abs(baseline_reward) * 100
                else:
                    reward_improvement = float('inf') if trained_reward > 0 else float('-inf')
                
                baseline_waiting = waiting_data[pattern][baseline_idx]
                trained_waiting = waiting_data[pattern][trained_idx]
                
                if baseline_waiting != 0:
                    waiting_improvement = (baseline_waiting - trained_waiting) / baseline_waiting * 100
                else:
                    waiting_improvement = float('inf') if trained_waiting < baseline_waiting else float('-inf')
                
                baseline_throughput = throughput_data[pattern][baseline_idx]
                trained_throughput = throughput_data[pattern][trained_idx]
                
                if baseline_throughput != 0:
                    throughput_improvement = (trained_throughput - baseline_throughput) / baseline_throughput * 100
                else:
                    throughput_improvement = float('inf') if trained_throughput > 0 else float('-inf')
                
                improvements[pattern] = {
                    "reward_improvement": float(reward_improvement),
                    "waiting_time_improvement": float(waiting_improvement),
                    "throughput_improvement": float(throughput_improvement)
                }
        
        # Save comparative analysis results
        comparative_results = {
            "agent_types": agent_types,
            "traffic_patterns": traffic_patterns,
            "rewards_data": rewards_data,
            "waiting_data": waiting_data,
            "throughput_data": throughput_data,
            "improvements": improvements
        }
        
        with open(os.path.join(save_dir, "comparative_analysis.json"), 'w') as f:
            json.dump(comparative_results, f, indent=4)
        
        logger.info(f"Comparative analysis completed and saved to {save_dir}")
        
        return comparative_results
    
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def analyze_decision_boundaries(agent_analysis, save_dir="results"):
    """
    Analyze and visualize agent decision boundaries.
    
    Args:
        agent_analysis: Dictionary of agent analysis results
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have decision boundaries data
        if "decision_boundaries" not in agent_analysis:
            logger.warning("No decision boundaries data found in agent analysis")
            return {}
        
        # Extract decision boundaries data
        boundaries = agent_analysis["decision_boundaries"]
        
        # Create visualizations for each light state
        for light_state, data in boundaries.items():
            if not data:
                continue
                
            # Extract data for visualization
            ns_densities = np.array([point["ns_density"] for point in data])
            ew_densities = np.array([point["ew_density"] for point in data])
            actions = np.array([point["action"] for point in data])
            
            # Create density grid
            grid_size = int(np.sqrt(len(data)))
            action_grid = actions.reshape(grid_size, grid_size)
            
            # Plot decision boundary
            plt.figure(figsize=(10, 8))
            
            # Custom colormap for the two actions: Green for NS=0, Red for EW=1
            cmap = ListedColormap(['green', 'red'])
            
            plt.imshow(
                action_grid, 
                origin='lower', 
                cmap=cmap,
                extent=[0, 1, 0, 1],
                aspect='auto'
            )
            
            # Add labels and title
            plt.xlabel('East-West Density')
            plt.ylabel('North-South Density')
            plt.title(f'Traffic Light Decision Policy (Current Light: {light_state})')
            
            # Add colorbar with labels
            cbar = plt.colorbar(ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['NS Green', 'EW Green'])
            
            # Add grid
            plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f"decision_boundary_{light_state}.png"))
            plt.close()
            
            # Calculate statistics
            ns_greater = np.sum(ns_densities > ew_densities)
            ew_greater = np.sum(ew_densities > ns_densities)
            ns_actions = np.sum(actions == 0)  # NS Green
            ew_actions = np.sum(actions == 1)  # EW Green
            
            correct_decisions = (
                np.sum((ns_densities > ew_densities) & (actions == 0)) +  # NS density higher and NS green
                np.sum((ew_densities > ns_densities) & (actions == 1))    # EW density higher and EW green
            )
            
            total_decisions = len(data)
            decision_accuracy = correct_decisions / total_decisions * 100 if total_decisions > 0 else 0
            
            # Save statistics
            stats = {
                "light_state": light_state,
                "ns_density_higher_count": int(ns_greater),
                "ew_density_higher_count": int(ew_greater),
                "ns_green_actions": int(ns_actions),
                "ew_green_actions": int(ew_actions),
                "correct_decisions": int(correct_decisions),
                "total_decisions": int(total_decisions),
                "decision_accuracy": float(decision_accuracy)
            }
            
            with open(os.path.join(save_dir, f"decision_stats_{light_state}.json"), 'w') as f:
                json.dump(stats, f, indent=4)
        
        logger.info(f"Decision boundary analysis completed and saved to {save_dir}")
        
        return {"decision_analysis_complete": True}
    
    except Exception as e:
        logger.error(f"Error analyzing decision boundaries: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def create_comprehensive_report(training_metrics=None, benchmark_results=None, agent_analysis=None, 
                              eval_results=None, save_dir="results/report"):
    """
    Create a comprehensive HTML report combining all analysis results.
    
    Args:
        training_metrics: Training metrics data (optional)
        benchmark_results: Benchmark results data (optional)
        agent_analysis: Agent analysis data (optional)
        eval_results: Evaluation results data (optional)
        save_dir: Directory to save the report
        
    Returns:
        Path to the HTML report
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Traffic RL Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                .plot-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
                .plot { margin-bottom: 20px; text-align: center; }
                .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .caption { font-style: italic; color: #666; margin-top: 5px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .highlight { background-color: #e6f7ff; font-weight: bold; }
                .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .card { flex: 1; min-width: 300px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .summary-stat { font-size: 24px; font-weight: bold; color: #0066cc; }
            </style>
        </head>
        <body>
            <h1>Traffic Light Control Reinforcement Learning Analysis</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        # 1. Training Results Section
        if training_metrics:
            html_content += """
            <div class="section">
                <h2>Training Results</h2>
            """
            
            # Extract key metrics
            rewards = training_metrics.get("rewards", [])
            avg_rewards = training_metrics.get("avg_rewards", [])
            training_time = training_metrics.get("training_time", 0)
            
            # Add summary stats
            html_content += """
                <div class="flex-container">
                    <div class="card">
                        <h3>Training Episodes</h3>
                        <div class="summary-stat">""" + str(len(rewards)) + """</div>
                    </div>
                    <div class="card">
                        <h3>Final Average Reward</h3>
                        <div class="summary-stat">""" + f"{np.mean(rewards[-100:]) if len(rewards) > 100 else np.mean(rewards):.2f}" + """</div>
                        <p>(Average of last 100 episodes)</p>
                    </div>
                    <div class="card">
                        <h3>Training Time</h3>
                        <div class="summary-stat">""" + f"{training_time:.2f}" + """</div>
                        <p>seconds</p>
                    </div>
                </div>
            """
            
            # Copy and include training plots
            plots = ["reward_plot.png", "loss_plot.png", "epsilon_plot.png", "performance_metrics.png"]
            html_content += """
                <h3>Training Metrics</h3>
                <div class="plot-container">
            """
            
            for plot in plots:
                src_path = os.path.join("results", plot)  # Assuming plots are in 'results' directory
                if os.path.exists(src_path):
                    # Copy plot to report directory
                    import shutil
                    shutil.copy2(src_path, os.path.join(plots_dir, plot))
                    
                    # Add to HTML
                    html_content += f"""
                    <div class="plot">
                        <img src="plots/{plot}" alt="{plot}">
                        <div class="caption">{plot.replace('_', ' ').replace('.png', '')}</div>
                    </div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        # 2. Benchmark Results Section
        if benchmark_results:
            html_content += """
            <div class="section">
                <h2>Benchmark Results</h2>
            """
            
            # Extract results if available
            if "results" in benchmark_results:
                results = benchmark_results["results"]
                
                # Create results table
                html_content += """
                <h3>Performance Comparison</h3>
                <table>
                    <tr>
                        <th>Agent & Pattern</th>
                        <th>Average Reward</th>
                        <th>Average Waiting Time</th>
                        <th>Average Throughput</th>
                    </tr>
                """
                
                for key, result in results.items():
                    html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{result.get('avg_reward', 0):.2f}</td>
                        <td>{result.get('avg_waiting_time', 0):.2f}</td>
                        <td>{result.get('avg_throughput', 0):.2f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Include benchmark plots
            plots = ["reward_comparison.png", "waiting_time_comparison.png", "throughput_comparison.png"]
            html_content += """
                <h3>Benchmark Comparisons</h3>
                <div class="plot-container">
            """
            
            for plot in plots:
                src_path = os.path.join("results/benchmark", plot)  # Assuming plots are in benchmark directory
                if os.path.exists(src_path):
                    # Copy plot to report directory
                    import shutil
                    shutil.copy2(src_path, os.path.join(plots_dir, plot))
                    
                    # Add to HTML
                    html_content += f"""
                    <div class="plot">
                        <img src="plots/{plot}" alt="{plot}">
                        <div class="caption">{plot.replace('_', ' ').replace('.png', '')}</div>
                    </div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        # 3. Agent Analysis Section
        if agent_analysis:
            html_content += """
            <div class="section">
                <h2>Agent Behavior Analysis</h2>
            """
            
            # Extract decision boundaries if available
            decision_boundaries = agent_analysis.get("decision_boundaries", {})
            
            if decision_boundaries:
                html_content += """
                <h3>Decision Boundary Analysis</h3>
                <p>Visualization of the agent's decision policy based on traffic densities.</p>
                <div class="plot-container">
                """
                
                # Add decision boundary plots
                for light_state in decision_boundaries.keys():
                    plot_file = f"decision_boundary_{light_state}.png"
                    src_path = os.path.join("results/analysis", plot_file)
                    
                    if os.path.exists(src_path):
                        # Copy plot to report directory
                        import shutil
                        shutil.copy2(src_path, os.path.join(plots_dir, plot_file))
                        
                        # Add to HTML
                        html_content += f"""
                        <div class="plot">
                            <img src="plots/{plot_file}" alt="Decision Boundary for Light State {light_state}">
                            <div class="caption">Decision policy for current light state: {light_state}</div>
                        </div>
                        """
                
                html_content += """
                </div>
                """
                
                # Add decision statistics
                html_content += """
                <h3>Decision Policy Statistics</h3>
                <table>
                    <tr>
                        <th>Light State</th>
                        <th>NS Actions</th>
                        <th>EW Actions</th>
                        <th>Correct Decisions</th>
                        <th>Decision Accuracy</th>
                    </tr>
                """
                
                for light_state in decision_boundaries.keys():
                    stats_file = os.path.join("results/analysis", f"decision_stats_{light_state}.json")
                    
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        html_content += f"""
                        <tr>
                            <td>{light_state}</td>
                            <td>{stats.get('ns_green_actions', 0)} ({stats.get('ns_green_actions', 0) / stats.get('total_decisions', 1) * 100:.1f}%)</td>
                            <td>{stats.get('ew_green_actions', 0)} ({stats.get('ew_green_actions', 0) / stats.get('total_decisions', 1) * 100:.1f}%)</td>
                            <td>{stats.get('correct_decisions', 0)} / {stats.get('total_decisions', 0)}</td>
                            <td>{stats.get('decision_accuracy', 0):.1f}%</td>
                        </tr>
                        """
                
                html_content += """
                </table>
                """
            
            html_content += """
            </div>
            """
        
        # 4. Evaluation Results Section
        if eval_results:
            html_content += """
            <div class="section">
                <h2>Evaluation Results</h2>
                <h3>Model Performance</h3>
                <table>
                    <tr>
                        <th>Traffic Pattern</th>
                        <th>Average Reward</th>
                        <th>Average Waiting Time</th>
                        <th>Average Throughput</th>
                    </tr>
            """
            
            html_content += f"""
                <tr>
                    <td>{eval_results.get('traffic_pattern', 'Unknown')}</td>
                    <td>{eval_results.get('avg_reward', 0):.2f} ± {eval_results.get('std_reward', 0):.2f}</td>
                    <td>{eval_results.get('avg_waiting_time', 0):.2f}</td>
                    <td>{eval_results.get('avg_throughput', 0):.2f}</td>
                </tr>
            """
            
            html_content += """
                </table>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        report_path = os.path.join(save_dir, "analysis_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report generated at {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error creating comprehensive report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None