import time
import json
from dataclasses import dataclass
from setfit import SetFitModel

nlp_model = SetFitModel.from_pretrained("malmoTextClassifier")

@dataclass
class ObjectPosition:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    id: str

    def teleport_str(self) -> str:
        return f"tp {self.x} {self.y} {self.z}"

def find_entity(agent_host, name: str, max_retries=5):
    position = None
    retries = 0
    while position is None and retries <= max_retries:
        retries += 1
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state == 0:
            time.sleep(0.5)
            continue
        latest_observation = json.loads(world_state.observations[-1].text)
        if DEBUG: print(latest_observation)
        if 'NearbyEntities' in latest_observation:
            for nearby_entity in latest_observation['NearbyEntities']:
                if DEBUG: print(nearby_entity)
                if nearby_entity['name'] == name:
                    if DEBUG: print(f"Matched horse entity: {nearby_entity}")
                    position = ObjectPosition(
                        x=nearby_entity['x'], 
                        y=nearby_entity['y'], 
                        z=nearby_entity['z'], 
                        yaw=nearby_entity['yaw'], 
                        pitch=nearby_entity['pitch'], 
                        id=nearby_entity['id']
                    )
    return position

def get_prediction(input_text):
    return nlp_model([input_text]).tolist()[0]

def task_0(agent_host):
    """
    Complete task of opening chest
    """
    print("Opening the chest")
    # Move straight towards chest
    agent_host.sendCommand("move 1")
    time.sleep(2.25)
    agent_host.sendCommand("move 0")

    # Look down
    time.sleep(2.10)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    # Open chest
    agent_host.sendCommand("use 1")
    time.sleep(2)

    if DEBUG:
        reset_agent(agent_host)


def task_1(agent_host):
    """
    Complete task of breaking flower
    """
    print("Breaking a flower")
    # Move straight towards being adjacent to flower
    agent_host.sendCommand("move 1")
    time.sleep(1.3)
    agent_host.sendCommand("move 0")

    # Turn to face flower
    agent_host.sendCommand("turn -1")
    time.sleep(0.5)
    agent_host.sendCommand("turn 0")

    # Move to get close to flower
    agent_host.sendCommand("move 1")
    time.sleep(1.0)
    agent_host.sendCommand("move 0")

    # Look down
    time.sleep(1.0)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.9)
    agent_host.sendCommand("pitch 0")

    # Break flower
    # agent_host.sendCommand("attack 1")
    # agent_host.sendCommand("attack 0")
    time.sleep(2)

    if DEBUG:
        reset_agent(agent_host)

def task_2(agent_host):
    """Go to horse"""
    print("Going to horse")
    position = find_entity(agent_host, 'Horse')
    if position is not None:
        if DEBUG: print(position.teleport_str())
        agent_host.sendCommand(position.teleport_str())
        agent_host.sendCommand("setYaw 0")
        agent_host.sendCommand("setPitch 0")
    else:
        print("Could not find a horse.")

def reset_agent(agent_host, teleport_x=0.5, teleport_z=0):
    """Directly teleport to spawn and reset direction agent is facing."""
    tp_command = "tp " + str(teleport_x) + " 227 " + str(teleport_z)
    agent_host.sendCommand(tp_command)
    agent_host.sendCommand("setYaw 0")
    agent_host.sendCommand("setPitch 0")