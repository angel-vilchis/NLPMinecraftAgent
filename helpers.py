import time
import json
from object_information import ObjectPosition, DEFAULT_OBJECTS, DrawObjectType
from setfit import SetFitModel
import transformers
import numpy as np

nlp_model = SetFitModel.from_pretrained("malmoTextClassifier")
text_gen = transformers.pipeline("text-generation", model="gpt2")
import warnings

label_names = {0: "Open chest", 
               1: "Smell plant",
               2: "Go to mob",
               3: "Jump in water", 
               4: "Sit next to campfire",
               5: "Play music",
               6: "Go through fence",
               7: "Go inside door",
               8: "Talk to user"}

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
    probs = nlp_model.predict_proba([input_text])[0].tolist()
    print("Probability prediction:\n", dict(zip(label_names.values(), np.round(probs, 2))))
    return np.argmax(probs)

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

def task_1(agent_host):
    """
    Complete task of smelling flower
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

def task_2(agent_host, input_text: str):
    """Go to entity"""
    entity = input_text.strip().split()[-1].capitalize()
    print(f"Going to {entity}")
    if entity not in DEFAULT_OBJECTS.keys():
        print(f"'{entity}' is not an entity in this environment.")
        print("Try one of these entities:")
        print([k for k, v in DEFAULT_OBJECTS.items() if v.draw_object_type == DrawObjectType.DRAW_ENTITY])
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
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["water1"].teleport_str(diff=(0.5, 0, -0.5)))

def task_4(agent_host):
    """Sit next to campfire"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["netherrack"].teleport_str(diff=(0.5, 0, -0.5)))

def task_5(agent_host):
    """Hit jukebox"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["jukebox"].teleport_str(diff=(0.5, 0, -0.5)))

    time.sleep(2.10)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

def task_6(agent_host):
    """Go through gate"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["fence_gate"].teleport_str(diff=(0.5, 0, -0.5)))
    go_through_entrance(agent_host)


def task_7(agent_host):
    """Go through door"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["wooden_door"].teleport_str(diff=(0.5, 0, -0.5)))
    go_through_entrance(agent_host)
    time.sleep(0.1)
    agent_host.sendCommand("turn -0.5")
    time.sleep(0.25)
    agent_host.sendCommand("turn 0")

    time.sleep(0.1)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

def task_8(agent_host):
    """Chat with agent"""
    user_prompt = "User: "
    agent_prompt = "\nMy long response: "

    chat_input = input("Say something to bot or 'quit'/'q' to stop chatting: ")
    while chat_input.lower() != "quit" and chat_input.lower() != "q":
        model_input = user_prompt + chat_input + agent_prompt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_output = text_gen(model_input, max_new_tokens=30, pad_token_id=50256, num_return_sequences=1)[0]["generated_text"]
        chat_text = model_output.split(":")[-1].strip()
        agent_host.sendCommand(f"chat {chat_text}")
        chat_input = input("Say something to bot or 'quit'/'q' to stop chatting: ")

def go_through_entrance(agent_host):
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")
    time.sleep(0.5)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

    time.sleep(0.5)
    agent_host.sendCommand("move 1")
    time.sleep(0.5)
    agent_host.sendCommand("move 0")

    agent_host.sendCommand("pitch -0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    time.sleep(0.1)
    agent_host.sendCommand("setYaw 180")
    time.sleep(0.2)

    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    time.sleep(0.5)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

def task_execution_print(task):
    print(f"Executing task: {label_names[task]}")

def reset_agent(agent_host, teleport_x=0.5, teleport_z=0, teleport_to_spawn=False):
    """Directly teleport to spawn and reset direction agent is facing."""
    if teleport_to_spawn:
        tp_command = "tp " + str(teleport_x) + " 227 " + str(teleport_z)
        agent_host.sendCommand(tp_command)
    agent_host.sendCommand("setYaw 0")
    agent_host.sendCommand("setPitch 0")