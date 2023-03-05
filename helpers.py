import time
import random
from setfit import SetFitModel, SetFitTrainer

nlp_model = SetFitModel.from_pretrained("malmoTextClassifier")

def get_prediction(input_text):
    return nlp_model([input_text]).tolist()[0]

def task_0(agent_host):
    """
        Complete task of opening chest
    """
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

def task_1(agent_host):
    """
        Complete task of breaking flower
    """
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
    agent_host.sendCommand("attack 1")
    agent_host.sendCommand("attack 0")
    time.sleep(2)

def reset_agent(agent_host, teleport_x=0.5, teleport_z=0):
        """Directly teleport to spawn and reset direction agent is facing."""
        tp_command = "tp " + str(teleport_x) + " 227 " + str(teleport_z)
        agent_host.sendCommand(tp_command)
        agent_host.sendCommand("setYaw 0")
        agent_host.sendCommand("setPitch 0")