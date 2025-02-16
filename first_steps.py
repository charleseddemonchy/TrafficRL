import os
import sys
import numpy as np
from collections import defaultdict
import traci
import traci.constants as tc
import matplotlib.pyplot as plt

# SUMO Path configuration
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class TrafficLightQLearning:
    def __init__(self, junction_id, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.junction_id = junction_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # Two actions: 0 = keep phase, 1 = change phase.
        self.q_table = defaultdict(lambda: np.zeros(2))
        
        # Retrieve allowed number of phases
        try:
            tls_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)
            if tls_def and len(tls_def) > 0:
                self.allowed_phases = len(tls_def[0].phases)
            else:
                self.allowed_phases = 3
        except Exception as e:
            print(f"Error retrieving phase definitions for {junction_id}: {e}")
            self.allowed_phases = 3

    def get_state(self):
        """Discretize the state based on lane waiting times, queue lengths, current phase, and phase duration."""
        try:
            incoming_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        except Exception as e:
            print(f"Error getting controlled lanes for junction {self.junction_id}: {e}")
            return None

        waiting_times = []
        queue_lengths = []
        for lane in incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
            waiting_times.append(wait_time)
            queue_lengths.append(traci.lane.getLastStepHaltingNumber(lane))
        
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        try:
            phase_duration = traci.trafficlight.getPhaseDuration(self.junction_id)
        except Exception:
            phase_duration = 0

        # Create a discrete state tuple
        state = (
            tuple(1 if w > 0 else 0 for w in waiting_times),
            tuple(1 if q > 3 else 0 for q in queue_lengths),
            current_phase,
            1 if phase_duration > 30 else 0
        )
        return state
    
    def get_reward(self):
        """
        Negative reward based on total waiting time and queue lengths.
        Queue length is weighted more heavily (×10) to strongly penalize jams.
        """
        incoming_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        total_waiting_time = 0
        total_queue_length = 0
        for lane in incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            total_waiting_time += sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        reward = -(total_waiting_time + total_queue_length * 10)
        return reward
    
    def choose_action(self, state):
        """Choose an action using an epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning update rule."""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

def run_simulation():
    # Configure SUMO binary and configuration file.
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    sumo_config = "paris.sumocfg"  # Adjust the path as needed.
    sumo_cmd = [sumo_binary, "-c", sumo_config]
    
    # Start SUMO with TraCI.
    traci.start(sumo_cmd)
    traci.simulationStep()  # Make sure the simulation has started.
    
    # Retrieve the list of traffic lights.
    tl_ids = traci.trafficlight.getIDList()
    if not tl_ids:
        print("No traffic lights found in the simulation.")
        traci.close()
        return
    
    # Select the first traffic light.
    junction_id = tl_ids[0]
    print(f"Using traffic light: {junction_id}")
    agent = TrafficLightQLearning(junction_id)
    print(f"Allowed phases for {junction_id}: {agent.allowed_phases}")
    
    # Set up real-time plotting of reward.
    plt.ion()  # Turn interactive mode on.
    fig, ax = plt.subplots()
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward over Simulation Steps")
    rewards_list = []
    steps_list = []
    line, = ax.plot(steps_list, rewards_list, 'b-')
    
    step = 0
    max_steps = 3600  # Run for 3600 simulation steps.
    
    while step < max_steps:
        traci.simulationStep()
        
        # Get the current state.
        current_state = agent.get_state()
        if current_state is None:
            break
        
        # Choose an action: 0 = keep phase, 1 = change phase.
        action = agent.choose_action(current_state)
        if action == 1:
            current_phase = traci.trafficlight.getPhase(junction_id)
            next_phase = (current_phase + 1) % agent.allowed_phases
            try:
                traci.trafficlight.setPhase(junction_id, next_phase)
            except Exception as e:
                print(f"Error setting phase from {current_phase} to {next_phase}: {e}")
        
        # Compute reward and next state.
        reward = agent.get_reward()
        next_state = agent.get_state()
        agent.learn(current_state, action, reward, next_state)
        
        # Log every 100 steps.
        if step % 100 == 0:
            print(f"Step {step:5d}, Reward: {reward:.2f}")
        
        # Append data for real-time plotting.
        steps_list.append(step)
        rewards_list.append(reward)
        
        # Update the plot every 50 steps.
        if step % 50 == 0:
            line.set_xdata(steps_list)
            line.set_ydata(rewards_list)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
            
        step += 1
    
    # Turn off interactive plotting and show the final plot.
    plt.ioff()
    plt.show()
    traci.close()

if __name__ == "__main__":
    run_simulation()
