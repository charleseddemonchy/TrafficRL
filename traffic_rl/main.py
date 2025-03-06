"""
Main Entry Point
==============
Main script to run training, evaluation, visualization, or analysis.
"""

import os
import argparse
import logging
import numpy as np
import torch
import random
import json

# Import configuration
from traffic_rl.config import load_config, override_config_with_args

# Import modules
from traffic_rl.train import train
from traffic_rl.evaluate import evaluate_agent
from traffic_rl.environment.traffic_simulation import TrafficSimulation
from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.utils.visualization import visualize_results, save_visualization


def setup_logging(log_file=None, debug=False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger("TrafficRL")
    logger.setLevel(level)
    
    # Debug message to confirm setup
    if debug:
        logger.debug("Debug logging enabled")
    
    return logger


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        logger.info(f"Random seed set to {seed}")


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Light Control with RL')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'visualize', 'benchmark', 'record', 'analyze'],
                        help='Mode to run (train, test, visualize, benchmark, record, analyze)')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to model file for test/visualize/analyze modes')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--record-video', type=str, default=None,
                        help='Record a video of the environment (in record mode)')
    parser.add_argument('--video-duration', type=int, default=30,
                        help='Duration of recorded video in seconds')
    parser.add_argument('--traffic-pattern', type=str, default='uniform',
                        choices=['uniform', 'rush_hour', 'weekend', 'random'],
                        help='Traffic pattern to use for testing and visualization')
    parser.add_argument('--log-file', type=str, default='logs/traffic_rl.log',
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.log_file, args.debug)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed if provided
    set_random_seed(config.get("random_seed", 42))
    
    # Create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create output directory: {e}")
    
    # Execute selected mode
    try:
        if args.mode == 'train':
            # Train agent
            logger.info("Starting training...")
            metrics = train(config, model_dir=os.path.join(args.output))
            
            # Visualize results
            logger.info("Training complete, visualizing results...")
            visualize_results(metrics["rewards"], metrics["avg_rewards"], 
                            save_path=os.path.join(args.output, "training_progress.png"))
            
            # Save metrics to file
            try:
                metrics_path = os.path.join(args.output, "training_metrics.json")
                with open(metrics_path, 'w') as f:
                    # Convert numpy values to native types
                    serializable_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                            serializable_metrics[key] = [float(v) for v in value]
                        elif isinstance(value, (np.integer, np.floating)):
                            serializable_metrics[key] = float(value)
                        else:
                            serializable_metrics[key] = value
                    
                    json.dump(serializable_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Failed to save training metrics: {e}")
        
        elif args.mode == 'test':
            # Test trained agent
            logger.info("Testing trained agent...")
            results = evaluate_agent(
                config, 
                args.model, 
                traffic_pattern=args.traffic_pattern,
                num_episodes=args.episodes or 10
            )
            
            # Save results
            results_path = os.path.join(args.output, f"test_results_{args.traffic_pattern}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Test results saved to {results_path}")
            
            # Print summary
            logger.info("Test Results Summary:")
            logger.info(f"  Traffic Pattern: {args.traffic_pattern}")
            logger.info(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"  Average Waiting Time: {results['avg_waiting_time']:.2f}")
            logger.info(f"  Average Throughput: {results['avg_throughput']:.2f}")
        
        elif args.mode == 'visualize':
            # Initialize environment with visualization
            logger.info("Initializing environment for visualization...")
            config["visualization"] = True
            
            env = TrafficSimulation(
                config=config,
                visualization=True,
                random_seed=config.get("random_seed", 42)
            )
            
            # Set traffic pattern
            env.traffic_pattern = args.traffic_pattern
            env.traffic_config = config["traffic_patterns"].get(
                args.traffic_pattern, config["traffic_patterns"]["uniform"]
            )
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Load model
            logger.info(f"Loading model from {args.model}...")
            if not agent.load(args.model):
                logger.error(f"Failed to load model from {args.model}")
                return
            
            # Run visualization
            logger.info("Starting visualization...")
            state, _ = env.reset()
            state = state.flatten()
            
            total_reward = 0
            for step in range(config.get("max_steps", 1000)):
                action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = next_state.flatten()
                
                state = next_state
                total_reward += reward
                
                # Log status periodically
                if step % 100 == 0:
                    logger.info(f"Step {step}/{config.get('max_steps', 1000)} - Total reward: {total_reward:.2f}")
                
                if terminated or truncated:
                    break
            
            logger.info(f"Visualization complete - Total reward: {total_reward:.2f}")
            
            # Close environment
            env.close()
        
        elif args.mode == 'record':
            # Record video of the environment
            logger.info("Recording video of the environment...")
            
            # Create the simulation environment
            env = TrafficSimulation(
                config=config,
                visualization=False,  # Don't need pygame visualization for recording
                random_seed=config.get("random_seed", 42)
            )
            
            # Set traffic pattern
            env.traffic_pattern = args.traffic_pattern
            env.traffic_config = config["traffic_patterns"].get(
                args.traffic_pattern, config["traffic_patterns"]["uniform"]
            )
            
            # Load agent if model path is provided
            if os.path.exists(args.model):
                logger.info(f"Loading model from {args.model} for recording...")
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = DQNAgent(state_size, action_size, config)
                
                # Load model
                if agent.load(args.model):
                    logger.info(f"Recording video with trained agent...")
                    env.recording_agent = agent
                else:
                    logger.warning(f"Failed to load model, will use random actions instead")
                    env.recording_agent = None
            else:
                logger.info(f"No model specified, recording video with random actions...")
                env.recording_agent = None
            
            # Determine the video file path
            video_file = args.record_video if args.record_video else os.path.join(
                args.output, f"traffic_simulation_{args.traffic_pattern}.mp4"
            )
            
            # Save the video
            success = save_visualization(
                env,
                filename=video_file,
                fps=30,
                duration=args.video_duration or 30
            )
            
            if success:
                logger.info(f"Video saved to {video_file}")
            else:
                logger.error(f"Failed to create video")
            
            # Close environment
            env.close()
        
        elif args.mode == 'benchmark':
            # Benchmark different agents
            logger.info("Running benchmark mode...")
            
            # Create output directory
            benchmark_dir = os.path.join(args.output, "benchmark")
            os.makedirs(benchmark_dir, exist_ok=True)
            
            # Define traffic patterns to benchmark
            patterns = ["uniform", "rush_hour", "weekend"]
            
            # Results container
            benchmark_results = {}
            
            # Test each pattern
            for pattern in patterns:
                logger.info(f"Benchmarking on {pattern} traffic pattern...")
                
                # Evaluate trained agent
                if os.path.exists(args.model):
                    trained_results = evaluate_agent(
                        config, 
                        args.model, 
                        traffic_pattern=pattern,
                        num_episodes=args.episodes or 5
                    )
                    benchmark_results[f"trained_{pattern}"] = trained_results
                
                # Evaluate fixed timing agent (baseline)
                # For the baseline, we'll implement a simple fixed-timing policy
                # This could be moved to a separate module in a full implementation
                class FixedTimingAgent:
                    def __init__(self, action_size):
                        self.action_size = action_size
                        self.current_phase = 0
                        self.phase_duration = 30  # Fixed phase duration
                        self.timer = 0
                    
                    def act(self, state, eval_mode=False):
                        # Change phase every phase_duration steps
                        if self.timer >= self.phase_duration:
                            self.current_phase = (self.current_phase + 1) % self.action_size
                            self.timer = 0
                        
                        self.timer += 1
                        return self.current_phase
                
                # Initialize environment
                env = TrafficSimulation(
                    config=config,
                    visualization=False,
                    random_seed=config.get("random_seed", 42)
                )
                
                # Set traffic pattern
                env.traffic_pattern = pattern
                env.traffic_config = config["traffic_patterns"].get(
                    pattern, config["traffic_patterns"]["uniform"]
                )
                
                # Run baseline evaluation
                baseline_rewards = []
                baseline_waiting = []
                baseline_throughput = []
                
                # Initialize fixed timing agent
                fixed_agent = FixedTimingAgent(env.action_space.n)
                
                # Evaluate
                for episode in range(args.episodes or 5):
                    state, _ = env.reset()
                    state = state.flatten()
                    total_reward = 0
                    episode_waiting = 0
                    episode_throughput = 0
                    
                    for step in range(config.get("max_steps", 1000)):
                        action = fixed_agent.act(state, eval_mode=True)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        next_state = next_state.flatten()
                        
                        state = next_state
                        total_reward += reward
                        
                        # Track metrics
                        episode_waiting += info.get('average_waiting_time', 0)
                        episode_throughput += info.get('total_cars_passed', 0)
                        
                        if terminated or truncated:
                            break
                    
                    # Store episode results
                    baseline_rewards.append(total_reward)
                    baseline_waiting.append(episode_waiting / (step + 1))
                    baseline_throughput.append(episode_throughput)
                
                # Calculate baseline statistics
                baseline_results = {
                    "avg_reward": float(np.mean(baseline_rewards)),
                    "std_reward": float(np.std(baseline_rewards)),
                    "min_reward": float(np.min(baseline_rewards)),
                    "max_reward": float(np.max(baseline_rewards)),
                    "avg_waiting_time": float(np.mean(baseline_waiting)),
                    "avg_throughput": float(np.mean(baseline_throughput)),
                    "traffic_pattern": pattern,
                    "agent": "fixed_timing"
                }
                
                benchmark_results[f"baseline_{pattern}"] = baseline_results
                
                # Close environment
                env.close()
            
            # Save benchmark results
            benchmark_file = os.path.join(benchmark_dir, "benchmark_results.json")
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=4)
            
            logger.info(f"Benchmark results saved to {benchmark_file}")
            
            # Print summary
            logger.info("Benchmark Summary:")
            for key, results in benchmark_results.items():
                logger.info(f"  {key}:")
                logger.info(f"    Reward: {results.get('avg_reward', 'N/A'):.2f} ± {results.get('std_reward', 'N/A'):.2f}")
                logger.info(f"    Waiting Time: {results.get('avg_waiting_time', 'N/A'):.2f}")
                logger.info(f"    Throughput: {results.get('avg_throughput', 'N/A'):.2f}")
        
        elif args.mode == 'analyze':
            # Analyze agent behavior
            logger.info("Running analysis mode...")
            
            # Create output directory for analysis
            analysis_dir = os.path.join(args.output, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Create environment
            env = TrafficSimulation(
                config=config,
                visualization=False,
                random_seed=config.get("random_seed", 42)
            )
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, config)
            
            # Load model
            if not os.path.exists(args.model):
                logger.error(f"Model file {args.model} not found")
                return
            
            logger.info(f"Loading model from {args.model}...")
            if not agent.load(args.model):
                logger.error(f"Failed to load model")
                return
            
            # Analysis results container
            analysis_results = {
                "model_path": args.model,
                "config": config,
                "action_distributions": {},
                "state_values": {},
                "performance_metrics": {}
            }
            
            # Run analysis for each traffic pattern
            for pattern in ["uniform", "rush_hour", "weekend"]:
                logger.info(f"Analyzing agent behavior on {pattern} traffic pattern...")
                
                # Set traffic pattern
                env.traffic_pattern = pattern
                env.traffic_config = config["traffic_patterns"].get(
                    pattern, config["traffic_patterns"]["uniform"]
                )
                
                # Collect action distribution and state values
                action_counts = [0, 0]  # For binary action space [NS_GREEN, EW_GREEN]
                state_values = []
                episode_rewards = []
                episode_waiting_times = []
                episode_throughputs = []
                
                # Run episodes
                for episode in range(args.episodes or 5):
                    state, _ = env.reset()
                    state_flat = state.flatten()
                    total_reward = 0
                    episode_waiting = 0
                    episode_throughput = 0
                    episode_actions = []
                    
                    for step in range(config.get("max_steps", 1000)):
                        # Get Q-values from agent
                        state_tensor = torch.tensor(state_flat, dtype=torch.float32).to(agent.device)
                        with torch.no_grad():
                            q_values = agent.local_network(state_tensor).cpu().numpy()
                        
                        # Record state values
                        state_values.append({
                            "step": step,
                            "episode": episode,
                            "q_values": q_values.tolist(),
                            "max_q": float(np.max(q_values)),
                            "q_diff": float(q_values[1] - q_values[0])
                        })
                        
                        # Select action
                        action = agent.act(state_flat, eval_mode=True)
                        episode_actions.append(int(action))
                        action_counts[action] += 1
                        
                        # Take action
                        next_state, reward, terminated, truncated, info = env.step(action)
                        next_state_flat = next_state.flatten()
                        
                        # Update metrics
                        total_reward += reward
                        episode_waiting += info.get('average_waiting_time', 0)
                        episode_throughput += info.get('total_cars_passed', 0)
                        
                        # Update state
                        state = next_state
                        state_flat = next_state_flat
                        
                        if terminated or truncated:
                            break
                    
                    # Record episode metrics
                    episode_rewards.append(total_reward)
                    episode_waiting_times.append(episode_waiting / (step + 1))
                    episode_throughputs.append(episode_throughput)
                    
                    # Log episode summary
                    logger.info(f"Episode {episode+1}/{args.episodes or 5} - "
                               f"Reward: {total_reward:.2f}, "
                               f"NS_GREEN: {episode_actions.count(0)}, "
                               f"EW_GREEN: {episode_actions.count(1)}")
                
                # Calculate action distribution
                total_actions = sum(action_counts)
                action_distribution = {
                    "NS_GREEN_count": action_counts[0],
                    "EW_GREEN_count": action_counts[1],
                    "NS_GREEN_percent": action_counts[0] / total_actions * 100 if total_actions > 0 else 0,
                    "EW_GREEN_percent": action_counts[1] / total_actions * 100 if total_actions > 0 else 0
                }
                
                # Calculate performance metrics
                performance_metrics = {
                    "avg_reward": float(np.mean(episode_rewards)),
                    "std_reward": float(np.std(episode_rewards)),
                    "avg_waiting_time": float(np.mean(episode_waiting_times)),
                    "avg_throughput": float(np.mean(episode_throughputs))
                }
                
                # Store results for this pattern
                analysis_results["action_distributions"][pattern] = action_distribution
                analysis_results["performance_metrics"][pattern] = performance_metrics
                
                # Only store a subset of state values to keep file size reasonable
                if len(state_values) > 1000:
                    state_values = random.sample(state_values, 1000)
                analysis_results["state_values"][pattern] = state_values
            
            # Save analysis results
            analysis_file = os.path.join(analysis_dir, "agent_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            
            logger.info(f"Analysis results saved to {analysis_file}")
            
            # Print summary
            logger.info("Analysis Summary:")
            for pattern, metrics in analysis_results["performance_metrics"].items():
                logger.info(f"  {pattern}:")
                logger.info(f"    Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
                logger.info(f"    Waiting Time: {metrics['avg_waiting_time']:.2f}")
                logger.info(f"    Throughput: {metrics['avg_throughput']:.2f}")
                
                dist = analysis_results["action_distributions"][pattern]
                logger.info(f"    Action Distribution: NS={dist['NS_GREEN_percent']:.1f}%, EW={dist['EW_GREEN_percent']:.1f}%")
            
            # Close environment
            env.close()
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
