"""
Visualization Script
=================
Script for creating visualizations of the traffic environment and agent performance.
"""

import os
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt

# Import modules
from traffic_rl.config import load_config
from traffic_rl.environment import TrafficSimulation
from traffic_rl.agents import DQNAgent
from traffic_rl.utils.visualization import save_visualization
from traffic_rl.utils.logging import setup_logging

logger = logging.getLogger("TrafficRL.Visualize")

def visualize_environment(config, model_path=None, output_dir="results/visualizations", 
                          traffic_pattern="uniform", duration=30, fps=30, 
                          video_filename=None):
    """
    Create a video visualization of the traffic environment.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model (if None, random actions will be used)
        output_dir: Directory to save visualizations
        traffic_pattern: Traffic pattern to visualize
        duration: Duration of the video in seconds
        fps: Frames per second for the video
        video_filename: Specific filename for the video (if None, auto-generated)
        
    Returns:
        Path to the saved video file, or None if unsuccessful
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        env = TrafficSimulation(
            config=config,
            visualization=False,  # We'll use our custom visualization
            random_seed=config.get("random_seed", 42)
        )
        
        # Set traffic pattern
        if traffic_pattern in config["traffic_patterns"]:
            pattern_config = config["traffic_patterns"][traffic_pattern]
            env.traffic_pattern = traffic_pattern
            env.traffic_config = pattern_config
            logger.info(f"Using traffic pattern: {traffic_pattern}")
        else:
            logger.warning(f"Traffic pattern {traffic_pattern} not found, using uniform")
            env.traffic_pattern = "uniform"
            env.traffic_config = config["traffic_patterns"]["uniform"]
        
        # Initialize agent if model path is provided
        if model_path and os.path.exists(model_path):
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Load model
            if agent.load(model_path):
                logger.info(f"Model loaded successfully from {model_path}")
                env.recording_agent = agent
            else:
                logger.warning(f"Failed to load model from {model_path}, will use random actions")
                env.recording_agent = None
        else:
            logger.info("No model provided, will use random actions")
            env.recording_agent = None
        
        # Determine video filename
        if video_filename is None:
            agent_str = "trained" if env.recording_agent is not None else "random"
            video_filename = f"traffic_simulation_{traffic_pattern}_{agent_str}.mp4"
        
        video_path = os.path.join(output_dir, video_filename)
        
        # Create visualization
        logger.info(f"Creating visualization video: {video_path}")
        success = save_visualization(
            env,
            filename=video_path,
            fps=fps,
            duration=duration
        )
        
        # Close environment
        env.close()
        
        if success:
            logger.info(f"Visualization video saved to {video_path}")
            return video_path
        else:
            logger.error("Failed to create visualization video")
            return None
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def visualize_training_metrics(metrics_file, output_dir="results/visualizations"):
    """
    Create visualizations of training metrics.
    
    Args:
        metrics_file: Path to the JSON file containing training metrics
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to the saved visualization files
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # List to store output file paths
        output_files = []
        
        # Extract metrics
        rewards = metrics.get("rewards", [])
        avg_rewards = metrics.get("avg_rewards", [])
        eval_rewards = metrics.get("eval_rewards", [])
        epsilon_values = metrics.get("epsilon_values", [])
        loss_values = metrics.get("loss_values", [])
        
        # Plot rewards
        if rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, alpha=0.6, label='Episode Reward')
            
            if avg_rewards:
                plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
            
            if eval_rewards:
                # Plot evaluation rewards at their corresponding episodes
                eval_frequency = metrics.get("eval_frequency", 20)
                eval_episodes = [i * eval_frequency for i in range(len(eval_rewards))]
                plt.plot(eval_episodes, eval_rewards, 'ro-', label='Evaluation Reward')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time')
            plt.legend()
            plt.grid(True)
            
            rewards_file = os.path.join(output_dir, "rewards.png")
            plt.savefig(rewards_file)
            plt.close()
            
            output_files.append(rewards_file)
            logger.info(f"Rewards plot saved to {rewards_file}")
        
        # Plot epsilon decay
        if epsilon_values:
            plt.figure(figsize=(12, 6))
            plt.plot(epsilon_values)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate (Epsilon) Over Time')
            plt.grid(True)
            
            epsilon_file = os.path.join(output_dir, "epsilon.png")
            plt.savefig(epsilon_file)
            plt.close()
            
            output_files.append(epsilon_file)
            logger.info(f"Epsilon plot saved to {epsilon_file}")
        
        # Plot loss values
        if loss_values:
            plt.figure(figsize=(12, 6))
            plt.plot(loss_values)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.grid(True)
            
            loss_file = os.path.join(output_dir, "loss.png")
            plt.savefig(loss_file)
            plt.close()
            
            output_files.append(loss_file)
            logger.info(f"Loss plot saved to {loss_file}")
        
        return output_files
    
    except Exception as e:
        logger.error(f"Error visualizing training metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def visualize_traffic_patterns(config, output_dir="results/visualizations"):
    """
    Create visualizations of different traffic patterns.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the saved visualization file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract traffic patterns
        patterns = config.get("traffic_patterns", {})
        
        if not patterns:
            logger.warning("No traffic patterns found in configuration")
            return None
        
        # Time of day (x-axis)
        time_of_day = np.linspace(0, 24, 1440)  # 24 hours with minute resolution
        normalized_time = time_of_day / 24.0
        
        # Plot traffic patterns
        plt.figure(figsize=(12, 6))
        
        for pattern_name, pattern_config in patterns.items():
            # Calculate traffic intensity based on pattern
            if pattern_name == "uniform":
                # Uniform pattern has constant arrival rate
                arrival_rate = pattern_config.get("arrival_rate", 0.03)
                variability = pattern_config.get("variability", 0.01)
                
                # Add some random noise for visualization
                np.random.seed(42)  # For reproducibility
                intensity = np.ones_like(normalized_time) * arrival_rate
                intensity += np.random.normal(0, variability, size=len(normalized_time))
                
            elif pattern_name == "rush_hour":
                # Rush hour pattern has morning and evening peaks
                morning_peak = pattern_config.get("morning_peak", 0.33) * 24  # Convert to hours
                evening_peak = pattern_config.get("evening_peak", 0.71) * 24  # Convert to hours
                peak_intensity = pattern_config.get("peak_intensity", 2.0)
                base_arrival = pattern_config.get("base_arrival", 0.03)
                
                # Calculate intensity based on time of day
                morning_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - morning_peak) / 1.5) ** 2)
                evening_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - evening_peak) / 1.5) ** 2)
                intensity = base_arrival * (1 + morning_factor + evening_factor)
                
            elif pattern_name == "weekend":
                # Weekend pattern has one midday peak
                midday_peak = pattern_config.get("midday_peak", 0.5) * 24  # Convert to hours
                peak_intensity = pattern_config.get("peak_intensity", 1.5)
                base_arrival = pattern_config.get("base_arrival", 0.02)
                
                # Calculate intensity based on time of day
                midday_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - midday_peak) / 3) ** 2)
                intensity = base_arrival * (1 + midday_factor)
                
            else:
                logger.warning(f"Unknown traffic pattern: {pattern_name}")
                continue
            
            # Plot this pattern
            plt.plot(time_of_day, intensity, label=pattern_name)
        
        # Add labels and title
        plt.xlabel('Time of Day (hours)')
        plt.ylabel('Traffic Intensity')
        plt.title('Traffic Patterns Over 24 Hours')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 24)
        plt.xticks(np.arange(0, 25, 3))
        
        # Add time labels (morning, noon, evening, night)
        plt.annotate('Morning', xy=(8, plt.ylim()[0]), xytext=(8, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Noon', xy=(12, plt.ylim()[0]), xytext=(12, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Evening', xy=(18, plt.ylim()[0]), xytext=(18, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Night', xy=(22, plt.ylim()[0]), xytext=(22, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        
        # Save the plot
        patterns_file = os.path.join(output_dir, "traffic_patterns.png")
        plt.savefig(patterns_file)
        plt.close()
        
        logger.info(f"Traffic patterns visualization saved to {patterns_file}")
        
        return patterns_file
    
    except Exception as e:
        logger.error(f"Error visualizing traffic patterns: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Light Control Visualization")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output", type=str, default="results/visualizations", help="Output directory")
    parser.add_argument("--traffic-pattern", type=str, default="uniform", 
                        choices=["uniform", "rush_hour", "weekend"],
                        help="Traffic pattern to visualize")
    parser.add_argument("--metrics", type=str, default=None, help="Path to training metrics JSON file")
    parser.add_argument("--duration", type=int, default=30, help="Duration of video in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--patterns", action="store_true", help="Visualize traffic patterns")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run visualizations based on arguments
    if args.patterns:
        # Visualize traffic patterns
        visualize_traffic_patterns(config, output_dir=args.output)
    
    if args.metrics:
        # Visualize training metrics
        visualize_training_metrics(args.metrics, output_dir=args.output)
    
    # Create environment visualization
    video_path = visualize_environment(
        config=config,
        model_path=args.model,
        output_dir=args.output,
        traffic_pattern=args.traffic_pattern,
        duration=args.duration,
        fps=args.fps
    )
    
    if video_path:
        print(f"\nVisualization complete! Video saved to: {video_path}")
    else:
        print("\nVisualization failed. Check logs for details.")