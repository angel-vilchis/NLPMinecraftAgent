import time
import json
from object_information import ObjectPosition, DEFAULT_OBJECTS
from setfit import SetFitModel

nlp_model = SetFitModel.from_pretrained("malmoTextClassifier")

def get_latest_world_observations(agent_host):
    world_state = agent_host.peekWorldState()
    latest_observation = json.loads(world_state.observations[-1].text)
    return latest_observation

def flush_world_observations(agent_host):
    agent_host.getWorldState()

def face_entity(agent_host, name: str, max_rotations=150):
    """Attempts to face entity. Returns True if successful, False othwerise."""
    rotations = 0
    while rotations <= max_rotations:
        latest_observation = get_latest_world_observations(agent_host)
        if 'LineOfSight' in latest_observation:
            if DEBUG: print(f"Line of sight observation:\n{latest_observation['LineOfSight']}\n")
            if latest_observation['LineOfSight']['type'] == name:
                agent_host.sendCommand("turn 0")
                break
        else:
            if DEBUG: print("Did not find line of sight")
        agent_host.sendCommand("turn 0.4")
        time.sleep(0.1)
        if DEBUG: print(f"Rotation {rotations}")
        rotations += 1
    else:
        agent_host.sendCommand("turn 0")
        return False
    return True

def find_entity(agent_host, name: str, max_retries=5):
    position = None
    retries = 0
    while position is None and retries <= max_retries:
        retries += 1
        latest_observation = get_latest_world_observations(agent_host)
        if DEBUG: print(f"Latest observation:\n{latest_observation}\n")
        if 'NearbyEntities' in latest_observation:
            for nearby_entity in latest_observation['NearbyEntities']:
                if DEBUG: print(f"Nearby Entity:\n{nearby_entity}\n")
                if nearby_entity['name'] == name:
                    if DEBUG: print(f"Matched {name} entity: {nearby_entity}\n")
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
    # Teleport
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["chest"].teleport_str(diff=(0.5, 0, -0.5)))

    # Look down
    time.sleep(2.10)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    # Open chest
    agent_host.sendCommand("use 1")

def task_1(agent_host):
    """
    Complete task of breaking flower
    """
    # Teleport
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["red_flower"].teleport_str())

    # Turn to face flower
    agent_host.sendCommand("turn -1")
    time.sleep(0.4)
    agent_host.sendCommand("turn 0")

    # Look down
    time.sleep(1.0)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.9)
    agent_host.sendCommand("pitch 0")

    # Break flower
    agent_host.sendCommand("attack 1")
    agent_host.sendCommand("attack 0")

def task_2(agent_host, input_text: str):
    """Go to entity"""
    entity = input_text.strip().split()[-1].capitalize()
    print(f"Going to {entity}")
    if entity not in DEFAULT_OBJECTS.keys():
        print(f"'{entity}' is not an entity in this environment.")
        return
    entity_specification = DEFAULT_OBJECTS[entity]
    position = find_entity(agent_host, entity)
    if position is not None:
        if DEBUG: print(position.teleport_str())
        agent_host.sendCommand(f"setPitch {entity_specification.find_with_yaw}")
        agent_host.sendCommand(position.teleport_str(diff=(2, 0, 2)))
        face_entity(agent_host, entity)
    else:
        print(f"Could not find a {entity}.")
    flush_world_observations(agent_host)

def task_3(agent_host):
    """Get in water"""
    pass

def task_4(agent_host):
    """Go next to campfire"""
    pass

def task_5(agent_host):
    """Hit jukebox"""
    pass

def task_6(agent_host):
    """Go through gate"""
    pass

def task_7(agent_host):
    """Go through door"""
    pass

def reset_agent(agent_host, teleport_x=0.5, teleport_z=0, teleport_to_spawn=False):
    """Directly teleport to spawn and reset direction agent is facing."""
    if teleport_to_spawn:
        tp_command = "tp " + str(teleport_x) + " 227 " + str(teleport_z)
        agent_host.sendCommand(tp_command)
    agent_host.sendCommand("setYaw 0")
    agent_host.sendCommand("setPitch 0")