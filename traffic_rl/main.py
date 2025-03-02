"""
Main entry point for the Traffic Light Management with Reinforcement Learning project.
"""

import os
import argparse
import json
import numpy as np
import torch
import random
import logging
import time

from traffic_rl.config import load_config, save_config, DEFAULT_CONFIG
from traffic_rl.utils import setup_logger, enable_debug_logging
from traffic_rl.environment import TrafficSimulation, visualize_results, save_visualization
from traffic_rl.agents import DQNAgent

# Initialize logger
logger = setup_logger("TrafficRL", logging.INFO)

# Set default random seed for reproducibility
RANDOM_SEED = 42

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def train(config, model_dir="models"):
    """
    Train the agent with improved monitoring and stability features.
    
    Args:
        config: Configuration dict
        model_dir: Directory to save models
    
    Returns:
        Dictionary of training history and metrics
    """
    # Create model directory if not exists
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        logger.error(f"Failed to create model directory: {e}")
        # Fallback to current directory
        model_dir = "."
    
    try:
        # Initialize environment
        env = TrafficSimulation(
            grid_size=config["grid_size"],
            max_cars=config["max_cars"],
            green_duration=config["green_duration"],
            yellow_duration=config["yellow_duration"],
            visualization=config["visualization"],
            random_seed=RANDOM_SEED
        )
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Initialize training metrics
        metrics = {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "loss_values": [],
            "epsilon_values": [],
            "learning_rates": [],
            "waiting_times": [],
            "throughput": [],
            "training_time": 0
        }
        
        # Initialize early stopping variables
        best_eval_reward = -float('inf')
        patience = config.get("early_stopping_patience", 100)
        patience_counter = 0
        
        # Initialize dynamic traffic pattern
        current_pattern = "uniform"  # Start with uniform pattern
        pattern_schedule = {
            0: "uniform",         # Start with uniform
            100: "rush_hour",     # Switch to rush hour after 100 episodes
            200: "weekend",       # Switch to weekend after 200 episodes
            300: "uniform"        # Back to uniform after 300 episodes
        }
        
        # Record training start time
        start_time = time.time()
        
        # Training loop
        for episode in range(1, config["num_episodes"] + 1):
            # Check if we need to switch traffic pattern
            if episode in pattern_schedule:
                current_pattern = pattern_schedule[episode]
                pattern_config = config["traffic_patterns"].get(current_pattern, config["traffic_patterns"]["uniform"])
                logger.info(f"Switching to {current_pattern} traffic pattern at episode {episode}")
                env.traffic_pattern = current_pattern
                env.traffic_config = pattern_config
            
            # Reset environment
            state, _ = env.reset()
            state = state.flatten()  # Flatten for NN input
            
            # Initialize episode variables
            total_reward = 0
            episode_steps = 0
            waiting_time = 0
            throughput = 0
            
            # Episode loop
            for step in range(config["max_steps"]):
                # Select action
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()  # Flatten for NN input
                
                # Apply reward clipping if enabled
                if config.get("clip_rewards", False):
                    reward = np.clip(reward, -10.0, 10.0)
                
                # Apply reward scaling if specified
                if "reward_scale" in config:
                    reward *= config["reward_scale"]
                
                # Store experience
                agent.step(state, action, reward, next_state, terminated)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                episode_steps += 1
                waiting_time += info.get('average_waiting_time', 0)
                throughput += info.get('total_cars_passed', 0)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Store rewards and compute averages
            metrics["rewards"].append(total_reward)
            
            # Calculate average reward over last 100 episodes (or fewer if we don't have 100 yet)
            window_size = min(100, len(metrics["rewards"]))
            avg_reward = np.mean(metrics["rewards"][-window_size:])
            metrics["avg_rewards"].append(avg_reward)
            
            # Calculate average waiting time and throughput for this episode
            avg_waiting_time = waiting_time / episode_steps if episode_steps > 0 else 0
            avg_throughput = throughput / episode_steps if episode_steps > 0 else 0
            metrics["waiting_times"].append(avg_waiting_time)
            metrics["throughput"].append(avg_throughput)
            
            # Record epsilon and learning rate
            metrics["epsilon_values"].append(agent.epsilon)
            current_lr = agent.optimizer.param_groups[0]['lr']
            metrics["learning_rates"].append(current_lr)
            
            # Record loss values if available
            if hasattr(agent, 'loss_history') and agent.loss_history:
                # Get average loss over this episode
                metrics["loss_values"].append(np.mean(agent.loss_history[-episode_steps:]) if episode_steps > 0 else 0)
            
            # Log progress
            logger.info(f"Episode {episode}/{config['num_episodes']} - "
                       f"Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.4f}, LR: {current_lr:.6f}, "
                       f"Traffic: {current_pattern}")
            
            # Evaluate the agent periodically
            if episode % config["eval_frequency"] == 0:
                logger.info(f"Evaluating agent at episode {episode}...")
                eval_reward = evaluate(agent, env, num_episodes=5)
                metrics["eval_rewards"].append(eval_reward)
                logger.info(f"Evaluation - Avg Reward: {eval_reward:.2f}")
                
                # Check for improvement and save model if improved
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    try:
                        model_path = os.path.join(model_dir, "best_model.pth")
                        agent.save(model_path)
                        logger.info(f"New best model saved with reward: {best_eval_reward:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} evaluations")
                    
                    # Apply early stopping if patience is exceeded
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {patience} evaluations without improvement")
                        break
            
            # Save model periodically
            if episode % config["save_frequency"] == 0:
                try:
                    model_path = os.path.join(model_dir, f"model_episode_{episode}.pth")
                    agent.save(model_path)
                    logger.info(f"Model checkpoint saved at episode {episode}")
                except Exception as e:
                    logger.error(f"Failed to save model checkpoint: {e}")
            
            # Early stopping if we've reached a good performance
            if avg_reward > config.get("early_stopping_reward", float('inf')):
                logger.info(f"Early stopping at episode {episode} - Reached target performance")
                break
        
        # Save final model
        try:
            model_path = os.path.join(model_dir, "final_model.pth")
            agent.save(model_path)
            logger.info("Final model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Record total training time
        metrics["training_time"] = time.time() - start_time
        logger.info(f"Total training time: {metrics['training_time']:.2f} seconds")
        
        # Close environment
        env.close()
        
        # Return metrics
        return metrics
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "error": str(e)
        }

def evaluate(agent, env, num_episodes=10):
    """
    Evaluate the agent without exploration.
    
    Args:
        agent: The agent to evaluate
        env: The environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Average reward over episodes
    """
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten for NN input
        total_reward = 0
        
        for step in range(1000):  # Max steps
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten for NN input
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards)

def test(config, model_path, traffic_pattern="uniform", visualization=False):
    """
    Test a trained agent.
    
    Args:
        config: Configuration dict
        model_path: Path to the model file
        traffic_pattern: Traffic pattern to use
        visualization: Whether to enable visualization
    
    Returns:
        Dictionary of test results
    """
    try:            
        # Initialize environment
        env = TrafficSimulation(
            grid_size=config["grid_size"],
            max_cars=config["max_cars"],
            green_duration=config["green_duration"],
            yellow_duration=config["yellow_duration"],
            visualization=config["visualization"],
            random_seed=RANDOM_SEED
        )
        
        # Ensure visualization is properly initialized if enabled
        if visualization and not hasattr(env, 'screen'):
            try:
                env._init_visualization()
                logger.info("Visualization initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize visualization: {e}")
                logger.warning("Continuing without visualization")
                visualization = False
                config["visualization"] = False
        
        # Set the traffic pattern
        if traffic_pattern in config["traffic_patterns"]:
            env.traffic_pattern = traffic_pattern
            env.traffic_config = config["traffic_patterns"][traffic_pattern]
            logger.info(f"Using {traffic_pattern} traffic pattern")
        else:
            logger.warning(f"Traffic pattern {traffic_pattern} not found, using default")
        
        # Get state and action sizes
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, config)
        
        # Load model
        if not agent.load(model_path):
            logger.error(f"Failed to load model from {model_path}")
            return {"success": False, "error": "Failed to load model"}
        
        # Run test episodes
        results = {
            "rewards": [],
            "waiting_times": [],
            "throughput": [],
            "success": True
        }
        
        for episode in range(10):  # Run 10 test episodes
            state, _ = env.reset()
            state = state.flatten()
            total_reward = 0
            episode_steps = 0
            waiting_time = 0
            throughput = 0
            
            for step in range(1000):  # Max steps
                # Process pygame events for visualization
                if visualization:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return results
                
                # Get action from agent
                action = agent.act(state, eval_mode=True)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state.flatten()
                
                # Update statistics
                state = next_state
                total_reward += reward
                episode_steps += 1
                waiting_time += info.get('average_waiting_time', 0)
                throughput += info.get('total_cars_passed', 0)
                
                # Explicitly render if visualization is enabled
                if visualization:
                    env.render()
                    # Add a short delay to make visualization more visible
                    if step % 10 == 0:  # Only delay every 10 steps to maintain performance
                        time.sleep(0.05)
                
                if terminated or truncated:
                    break
            
            # Record results
            results["rewards"].append(total_reward)
            results["waiting_times"].append(waiting_time / episode_steps if episode_steps > 0 else 0)
            results["throughput"].append(throughput / episode_steps if episode_steps > 0 else 0)
            
            logger.info(f"Test Episode {episode+1}/10 - Reward: {total_reward:.2f}")
        
        # Calculate summary stats
        results["avg_reward"] = np.mean(results["rewards"])
        results["avg_waiting_time"] = np.mean(results["waiting_times"])
        results["avg_throughput"] = np.mean(results["throughput"])
        
        logger.info(f"Test Results - Avg Reward: {results['avg_reward']:.2f}, "
                  f"Avg Waiting Time: {results['avg_waiting_time']:.2f}, "
                  f"Avg Throughput: {results['avg_throughput']:.2f}")
        
        # Close environment
        if visualization:
            pygame.quit()
        env.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def record_video(config, model_path=None, output_path="traffic_simulation.mp4", 
                traffic_pattern="uniform", duration=30):
    """
    Record a video of the traffic simulation.
    
    Args:
        config: Configuration dict
        model_path: Path to the model file (optional)
        output_path: Path to save the video
        traffic_pattern: Traffic pattern to use
        duration: Duration of the video in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize environment
        env = TrafficSimulation(
            grid_size=config["grid_size"],
            max_cars=config["max_cars"],
            green_duration=config["green_duration"],
            yellow_duration=config["yellow_duration"],
            visualization=config["visualization"],
            random_seed=RANDOM_SEED
        )
        
        # Set the traffic pattern
        if traffic_pattern in config["traffic_patterns"]:
            env.traffic_pattern = traffic_pattern
            env.traffic_config = config["traffic_patterns"][traffic_pattern]
            logger.info(f"Using {traffic_pattern} traffic pattern")
        else:
            logger.warning(f"Traffic pattern {traffic_pattern} not found, using default")
        
        # Load agent if model path is provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            
            state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_size = env.action_space.n
            
            agent = DQNAgent(state_size, action_size, config)
            
            if agent.load(model_path):
                logger.info("Model loaded successfully")
                env.recording_agent = agent
            else:
                logger.warning("Failed to load model, using random actions")
                env.recording_agent = None
        else:
            logger.info("No model specified, using random actions")
            env.recording_agent = None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Record the video
        success = save_visualization(
            env,
            filename=output_path,
            fps=30,
            duration=duration
        )
        
        if success:
            logger.info(f"Video saved to {output_path}")
        else:
            logger.error("Failed to save video")
        
        # Close environment
        env.close()
        
        return success
        
    except Exception as e:
        logger.error(f"Video recording failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Traffic Light Management with Reinforcement Learning')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'visualize', 'record'],
                        help='Operation mode')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file for test/visualize/record modes')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--output', type=str, default='runs',
                        help='Output directory for results and models')
    
    # Environment parameters
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Size of the traffic grid')
    parser.add_argument('--traffic-pattern', type=str, default='uniform',
                        choices=['uniform', 'rush_hour', 'weekend'],
                        help='Traffic pattern to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    # Video recording
    parser.add_argument('--video-path', type=str, default='traffic_simulation.mp4',
                        help='Path to save the video (in record mode)')
    parser.add_argument('--video-duration', type=int, default=30,
                        help='Duration of the video in seconds')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        enable_debug_logging(logger)
        logger.debug("Debug mode enabled")
    
    # Set random seed
    if args.seed is not None:
        global RANDOM_SEED
        RANDOM_SEED = args.seed
        set_seed(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        config["num_episodes"] = args.episodes
    if args.grid_size:
        config["grid_size"] = args.grid_size
    if args.visualize:
        config["visualization"] = True
    
    # Save current configuration for reference
    save_config(config, os.path.join(args.output, 'config.json'))
    
    # Execute selected mode
    if args.mode == 'train':
        logger.info("Starting training mode...")
        
        # Run training
        metrics = train(config, model_dir=os.path.join(args.output, 'models'))
        
        # Visualize and save results
        if metrics.get("rewards"):
            # Generate and save training plots
            visualize_results(
                metrics["rewards"], 
                metrics["avg_rewards"],
                save_path=os.path.join(args.output, "training_progress.png")
            )
            
            # Save metrics as CSV
            try:
                results_path = os.path.join(args.output, "training_results.csv")
                with open(results_path, 'w') as f:
                    # Create header row
                    header = "episode,reward,avg_reward"
                    if metrics.get("loss_values"):
                        header += ",loss"
                    if metrics.get("epsilon_values"):
                        header += ",epsilon"
                    if metrics.get("learning_rates"):
                        header += ",learning_rate"
                    f.write(header + "\n")
                    
                    # Add data rows
                    for i in range(len(metrics["rewards"])):
                        row = f"{i+1},{metrics['rewards'][i]},{metrics['avg_rewards'][i]}"
                        if metrics.get("loss_values") and i < len(metrics["loss_values"]):
                            row += f",{metrics['loss_values'][i]}"
                        if metrics.get("epsilon_values") and i < len(metrics["epsilon_values"]):
                            row += f",{metrics['epsilon_values'][i]}"
                        if metrics.get("learning_rates") and i < len(metrics["learning_rates"]):
                            row += f",{metrics['learning_rates'][i]}"
                        f.write(row + "\n")
                logger.info(f"Training results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save CSV results: {e}")
        else:
            logger.error("Training did not produce valid results")
    
    elif args.mode == 'test':
        logger.info("Starting test mode...")
        
        if not args.model:
            # Try to use the best model by default
            model_path = os.path.join(args.output, 'models', 'best_model.pth')
            if not os.path.exists(model_path):
                # Try final model as fallback
                model_path = os.path.join(args.output, 'models', 'final_model.pth')
            if not os.path.exists(model_path):
                logger.error("No model specified and no default models found")
                return
        else:
            model_path = args.model
        
        logger.info(f"Testing model: {model_path}")
        
        # Run test
        results = test(
            config, 
            model_path, 
            traffic_pattern=args.traffic_pattern,
            visualization=args.visualize
        )
        
        # Save test results
        if results.get("success", False):
            try:
                results_path = os.path.join(args.output, "test_results.json")
                with open(results_path, 'w') as f:
                    # Convert numpy values to regular Python types for JSON
                    clean_results = {k: float(v) if isinstance(v, np.number) else v 
                                    for k, v in results.items() if k != "rewards"}
                    clean_results["rewards"] = [float(r) for r in results.get("rewards", [])]
                    
                    json.dump(clean_results, f, indent=4)
                logger.info(f"Test results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save test results: {e}")
        else:
            logger.error("Testing failed")
    
    elif args.mode == 'visualize':
        logger.info("Starting visualization mode...")
        
        # This mode is similar to test but with visualization enabled
        if not args.model:
            # Try to use the best model by default
            model_path = os.path.join(args.output, 'models', 'best_model.pth')
            if not os.path.exists(model_path):
                # Try final model as fallback
                model_path = os.path.join(args.output, 'models', 'final_model.pth')
            if not os.path.exists(model_path):
                logger.error("No model specified and no default models found")
                return
        else:
            model_path = args.model
        
        # Force visualization on
        config["visualization"] = True
        
        # Run visualization (same as test but with visualization enabled)
        results = test(
            config, 
            model_path, 
            traffic_pattern=args.traffic_pattern,
            visualization=True
        )
        
        if not results.get("success", False):
            logger.error("Visualization failed")
    
    elif args.mode == 'record':
        logger.info("Starting video recording mode...")
        
        model_path = None
        if args.model:
            model_path = args.model
        elif os.path.exists(os.path.join(args.output, 'models', 'best_model.pth')):
            model_path = os.path.join(args.output, 'models', 'best_model.pth')
        elif os.path.exists(os.path.join(args.output, 'models', 'final_model.pth')):
            model_path = os.path.join(args.output, 'models', 'final_model.pth')
        
        if model_path:
            logger.info(f"Using model: {model_path}")
        else:
            logger.info("No model found, will use random actions")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        
        # Record video
        success = record_video(
            config,
            model_path=model_path,
            output_path=args.video_path,
            traffic_pattern=args.traffic_pattern,
            duration=args.video_duration
        )
        
        if success:
            logger.info(f"Video recording saved to {args.video_path}")
        else:
            logger.error("Video recording failed")

if __name__ == "__main__":
    main()