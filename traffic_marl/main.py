"""
Main module for Traffic Light Management with Reinforcement Learning.
"""
import os
import sys
import argparse
import json
import logging
import numpy as np
import random
import torch
import time

from .environment import TrafficSimulation
from .agents import DQNAgent, FixedTimingAgent
from .training import train, evaluate, comparative_analysis
from .utils import setup_logging, enable_debug_logging, visualize_results, CONFIG, RANDOM_SEED

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Traffic Light Control with RL')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'visualize', 'benchmark', 'record', 'analyze'],
                        help='Mode to run (train, test, visualize, benchmark, record, analyze)')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to model file for test/visualize/analyze modes')
    parser.add_argument('--config', type=str, default='config.json',
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
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger("TrafficRL.Main")
    
    # Enable debug logging if requested
    if args.debug:
        enable_debug_logging()
        logger.debug("Debug mode enabled")
    
    # Create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create output directory: {e}")
    
    # Set random seed if provided
    if args.seed is not None:
        global RANDOM_SEED
        RANDOM_SEED = args.seed
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        logger.info(f"Using random seed: {RANDOM_SEED}")
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                # Update default config with loaded values
                CONFIG.update(loaded_config)
                logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file {args.config} not found, using defaults")
            # Save default config for reference
            with open(os.path.join(args.output, 'default_config.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Continuing with default configuration")
    
    # Override config with command line arguments
    if args.episodes:
        CONFIG["num_episodes"] = args.episodes
    
    if args.visualize:
        CONFIG["visualization"] = True
    
    # Log configuration
    logger.info(f"Running with configuration: {CONFIG}")
    
    try:
        if args.mode == 'train':
            # Train agent
            logger.info("Starting training...")
            metrics = train(CONFIG, model_dir=os.path.join(args.output, 'models'))
            
            # Visualize results
            logger.info("Training complete, visualizing results...")
            visualize_results(metrics["rewards"], metrics["avg_rewards"], 
                            save_path=os.path.join(args.output, "training_progress.png"))
            
            # Save results as csv
            try:
                results_path = os.path.join(args.output, "training_results.csv")
                with open(results_path, 'w') as f:
                    f.write("episode,reward,avg_reward")
                    if "loss_values" in metrics and metrics["loss_values"]:
                        f.write(",loss")
                    if "epsilon_values" in metrics:
                        f.write(",epsilon")
                    if "learning_rates" in metrics:
                        f.write(",learning_rate")
                    f.write("\n")
                    
                    for i in range(len(metrics["rewards"])):
                        line = f"{i},{metrics['rewards'][i]},{metrics['avg_rewards'][i]}"
                        if "loss_values" in metrics and i < len(metrics["loss_values"]):
                            line += f",{metrics['loss_values'][i]}"
                        if "epsilon_values" in metrics and i < len(metrics["epsilon_values"]):
                            line += f",{metrics['epsilon_values'][i]}"
                        if "learning_rates" in metrics and i < len(metrics["learning_rates"]):
                            line += f",{metrics['learning_rates'][i]}"
                        f.write(line + "\n")
                logger.info(f"Training results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save training results: {e}")
        
        elif args.mode == 'test':
            # Initialize environment
            logger.info("Initializing environment for testing...")
            
            # Configure traffic pattern
            traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
            logger.info(f"Using traffic pattern: {args.traffic_pattern}")
            
            env = TrafficSimulation(
                grid_size=CONFIG["grid_size"],
                max_cars=CONFIG["max_cars"],
                green_duration=CONFIG["green_duration"],
                yellow_duration=CONFIG["yellow_duration"],
                visualization=CONFIG["visualization"],
                random_seed=RANDOM_SEED
            )
            
            # Set the traffic pattern
            env.traffic_pattern = args.traffic_pattern
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize agent
            agent = DQNAgent(state_size, action_size, CONFIG)
            
            # Load model
            logger.info(f"Loading model from {args.model}...")
            agent.load(args.model)
            
            # Evaluate agent
            logger.info("Evaluating agent...")
            reward = evaluate(agent, env, num_episodes=10)
            logger.info(f"Evaluation - Avg Reward: {reward:.2f}")
            
            # Save evaluation results
            try:
                with open(os.path.join(args.output, "evaluation_results.json"), 'w') as f:
                    json.dump({
                        "average_reward": float(reward),
                        "traffic_pattern": args.traffic_pattern,
                        "model_path": args.model
                    }, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")
            
            # Close environment
            env.close()
        
        elif args.mode == 'visualize':
            # Initialize environment with visualization
            logger.info("Initializing environment for visualization...")
            CONFIG["visualization"] = True
            
            # Configure traffic pattern
            traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
            logger.info(f"Using traffic pattern: {args.traffic_pattern}")
            
            try:
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=True,
                    random_seed=RANDOM_SEED
                )
                
                # Set the traffic pattern
                env.traffic_pattern = args.traffic_pattern
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = DQNAgent(state_size, action_size, CONFIG)
                
                # Load model
                logger.info(f"Loading model from {args.model}...")
                agent.load(args.model)
                
                # Run visualization
                logger.info("Starting visualization...")
                state, _ = env.reset()
                state = state.flatten()
                
                total_reward = 0
                for step in range(1000):
                    action = agent.act(state, eval_mode=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = next_state.flatten()
                    
                    state = next_state
                    total_reward += reward
                    
                    # Log status periodically
                    if step % 100 == 0:
                        logger.info(f"Step {step}/1000 - Total reward: {total_reward:.2f}")
                    
                    # Add delay for visualization
                    try:
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        logger.info("Visualization interrupted by user")
                        break
                    
                    if terminated or truncated:
                        break
                
                logger.info(f"Visualization complete - Total reward: {total_reward:.2f}")
                
                # Close environment
                env.close()
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
        
        elif args.mode == 'benchmark':
            # Benchmark different configurations
            logger.info("Running benchmark mode...")
            
            # Initialize environment
            env = TrafficSimulation(
                grid_size=CONFIG["grid_size"],
                max_cars=CONFIG["max_cars"],
                green_duration=CONFIG["green_duration"],
                yellow_duration=CONFIG["yellow_duration"],
                visualization=False,
                random_seed=RANDOM_SEED
            )
            
            # Get state and action sizes
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            # Initialize RL agent
            rl_agent = DQNAgent(state_size, action_size, CONFIG)
            
            # Load model if exists
            if os.path.exists(args.model):
                logger.info(f"Loading model from {args.model}...")
                rl_agent.load(args.model)
            else:
                logger.warning(f"Model {args.model} not found, using untrained agent")
            
            # Initialize fixed timing agent
            fixed_agent = FixedTimingAgent(action_size)
            
            # Benchmark agents
            agents = [rl_agent, fixed_agent]
            labels = ["RL Agent", "Fixed Timing"]
            
            logger.info("Starting comparative analysis...")
            results = comparative_analysis(env, agents, labels, num_episodes=20)
            
            # Save benchmark results
            try:
                with open(os.path.join(args.output, "benchmark_results.json"), 'w') as f:
                    # Convert numpy values to native Python types for JSON serialization
                    clean_results = {}
                    for k, v in results.items():
                        if k == 'summary':
                            clean_summary = {}
                            for agent, metrics in v.items():
                                clean_metrics = {mk: float(mv) for mk, mv in metrics.items()}
                                clean_summary[agent] = clean_metrics
                            clean_results[k] = clean_summary
                        else:
                            clean_results[k] = v
                    
                    json.dump(clean_results, f, indent=4)
                logger.info(f"Benchmark results saved to {os.path.join(args.output, 'benchmark_results.json')}")
            except Exception as e:
                logger.error(f"Failed to save benchmark results: {e}")
            
            # Log summary results
            logger.info("Benchmark Summary:")
            for label, metrics in results['summary'].items():
                logger.info(f"  {label}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value:.4f}")
            
            # Close environment
            env.close()
        
        elif args.mode == 'record':
            # Video recording mode - create a video of the environment
            logger.info("Initializing environment for video recording...")
            
            try:
                # Configure environment based on selected traffic pattern
                traffic_config = CONFIG["traffic_patterns"].get(args.traffic_pattern, CONFIG["traffic_patterns"]["uniform"])
                logger.info(f"Using traffic pattern: {args.traffic_pattern}")
                
                # Initialize the environment
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=False,  # Don't need pygame visualization for our enhanced version
                    random_seed=RANDOM_SEED
                )
                
                # Set the traffic pattern for the environment
                env.traffic_pattern = args.traffic_pattern
                
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Determine the video filename
                video_file = args.record_video if args.record_video else f"{args.output}/traffic_simulation_{args.traffic_pattern}.mp4"
                
                # Create a video with either random or trained agent actions
                if os.path.exists(args.model):
                    logger.info(f"Loading model from {args.model} for recording...")
                    
                    # Get state and action sizes
                    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                    action_size = env.action_space.n
                    
                    # Initialize agent
                    agent = DQNAgent(state_size, action_size, CONFIG)
                    
                    # Load model
                    if agent.load(args.model):
                        logger.info(f"Recording video with trained agent...")
                        # Add agent as a parameter to the environment for video generation
                        env.recording_agent = agent
                    else:
                        logger.warning(f"Failed to load model, will use random actions instead")
                        env.recording_agent = None
                else:
                    logger.info(f"No model specified, recording video with random actions...")
                    env.recording_agent = None
                
                # Save the video
                #Change from env.save_visualization to env.save_visualization
                success = env.save_visualization(
                    filename=video_file,
                    fps=30,
                    duration=args.video_duration if args.video_duration else 30
                )
                
                if success:
                    logger.info(f"Video saved to {video_file}")
                else:
                    logger.error(f"Failed to create video")
                
                # Close the environment
                env.close()
                
            except Exception as e:
                logger.error(f"Error recording video: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        elif args.mode == 'analyze':
            # Analysis mode - analyze agent behavior and performance
            logger.info("Starting analysis mode...")
            
            try:
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Initialize environment
                env = TrafficSimulation(
                    grid_size=CONFIG["grid_size"],
                    max_cars=CONFIG["max_cars"],
                    green_duration=CONFIG["green_duration"],
                    yellow_duration=CONFIG["yellow_duration"],
                    visualization=True,
                    random_seed=RANDOM_SEED
                )
                
                # Get state and action sizes
                state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
                action_size = env.action_space.n
                
                # Initialize agent
                agent = DQNAgent(state_size, action_size, CONFIG)
                
                # Initialize analysis results
                analysis_results = {
                    "model_path": args.model,
                    "config": CONFIG,
                    "state_action_values": {},
                    "performance_metrics": {},
                    "decision_boundaries": {}
                }
                
                # Load model
                if not os.path.exists(args.model):
                    logger.error(f"Model file {args.model} not found")
                    return
                
                logger.info(f"Loading model from {args.model}...")
                agent.load(args.model)
                
                # Analyze trained policy
                logger.info("Analyzing trained policy...")
                
                # 1. Generate a grid of sample states to analyze
                logger.info("Generating sample states for analysis...")
                sample_states = []
                
                # Create states with varying traffic densities
                density_values = [0.1, 0.3, 0.5, 0.7, 0.9]
                for ns_density in density_values:
                    for ew_density in density_values:
                        for light_state in [0, 1]:  # Current light state
                            # Create a state with uniform density across all intersections
                            state = np.zeros((env.num_intersections, 3), dtype=np.float32)
                            state[:, 0] = ns_density  # NS density
                            state[:, 1] = ew_density  # EW density
                            state[:, 2] = light_state  # Light state
                            
                            sample_states.append((state, f"NS={ns_density:.1f},EW={ew_density:.1f},Light={light_state}"))
                
                # 2. Evaluate Q-values for each state
                logger.info("Evaluating Q-values for sample states...")
                q_values = {}
                
                for state, state_desc in sample_states:
                    # Flatten and convert state to tensor
                    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).to(agent.device)
                    
                    # Get Q-values from local network
                    agent.local_network.eval()
                    with torch.no_grad():
                        q_vals = agent.local_network(state_tensor).cpu().numpy()
                    
                    # Store Q-values and resulting actions
                    q_values[state_desc] = {
                        'q_values': q_vals.tolist(),
                        'action': int(np.argmax(q_vals)),
                        'q_diff': float(q_vals[1] - q_vals[0])  # Difference between actions
                    }
                
                analysis_results["state_action_values"] = q_values
                
                # 3. Test agent performance on different traffic patterns
                logger.info("Evaluating agent performance on different traffic patterns...")
                performance = {}
                
                for pattern in ["uniform", "rush_hour", "weekend"]:
                    if pattern in CONFIG["traffic_patterns"]:
                        logger.info(f"Testing on {pattern} traffic pattern...")
                        
                        # Set the traffic pattern
                        env.traffic_pattern = pattern
                        
                        # Run evaluation episodes
                        rewards = []
                        waiting_times = []
                        cars_passed = []
                        avg_densities = []
                        
                        for episode in range(10):  # Run 10 episodes per pattern
                            state, _ = env.reset()
                            state = state.flatten()
                            episode_reward = 0
                            episode_waiting = 0
                            episode_cars = 0
                            episode_density = []
                            
                            for step in range(100):  # Run 100 steps per episode
                                action = agent.act(state, eval_mode=True)
                                next_state, reward, terminated, truncated, info = env.step(action)
                                next_state = next_state.flatten()
                                
                                state = next_state
                                episode_reward += reward
                                episode_waiting += info['average_waiting_time']
                                episode_cars += info['total_cars_passed']
                                episode_density.append(info['traffic_density'])
                                
                                if terminated or truncated:
                                    break
                            
                            rewards.append(episode_reward)
                            waiting_times.append(episode_waiting / (step + 1))  # Average per step
                            cars_passed.append(episode_cars)
                            avg_densities.append(np.mean(episode_density))
                        
                        # Store results
                        performance[pattern] = {
                            'avg_reward': float(np.mean(rewards)),
                            'std_reward': float(np.std(rewards)),
                            'avg_waiting_time': float(np.mean(waiting_times)),
                            'avg_cars_passed': float(np.mean(cars_passed)),
                            'avg_density': float(np.mean(avg_densities))
                        }
                
                analysis_results["performance_metrics"] = performance
                
                # 4. Generate decision boundary data
                logger.info("Generating decision boundary analysis...")
                
                # Create a grid of NS vs EW densities
                ns_densities = np.linspace(0, 1, 20)
                ew_densities = np.linspace(0, 1, 20)
                
                # For each light state
                for light_state in [0, 1]:
                    decision_data = []
                    
                    for ns in ns_densities:
                        for ew in ew_densities:
                            # Create a simple state with one intersection
                            state = np.zeros((1, 3), dtype=np.float32)
                            state[0, 0] = ns  # NS density
                            state[0, 1] = ew  # EW density
                            state[0, 2] = light_state  # Light state
                            
                            # Get action from agent
                            flat_state = state.flatten()
                            action = agent.act(flat_state, eval_mode=True)
                            
                            # Store result
                            decision_data.append({
                                'ns_density': float(ns),
                                'ew_density': float(ew),
                                'action': int(action)
                            })
                    
                    analysis_results["decision_boundaries"][f"light_{light_state}"] = decision_data
                
                # 5. Save analysis results
                logger.info("Saving analysis results...")
                results_file = f"{args.output}/agent_analysis.json"
                
                with open(results_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                logger.info(f"Analysis results saved to {results_file}")
                
                # Close environment
                env.close()
                
            except Exception as e:
                logger.error(f"Error in analysis mode: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()