from __future__ import print_function

from builtins import range
try:
    from malmo import MalmoPython
except (ImportError, ModuleNotFoundError):
    print("Error importing MalmoPython from malmo. Trying with the local file.")
    try:
        import MalmoPython
    except (ImportError, ModuleNotFoundError):
        print("Error importing MalmoPython from local file.")
from pathlib import Path
import os
import sys
import time
import argparse

import helpers
parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_true', default=False)
args = parser.parse_args()
helpers.DEBUG = args.debug


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>NLP Action Classification from Text</Summary>
              </About>
              
              <ServerSection>
                <ServerInitialConditions>
                    <Time>
                      <StartTime>12000</StartTime>
                      <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                  </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" forceReset="true"/>
                  <DrawingDecorator>
                    <DrawBlock x="0" y="227" z="10" type="chest"/>
                    <DrawBlock x="5" y="227" z="5" type="red_flower"/>
                    <DrawEntity x="15" y="227" z="15" type="Horse"/>
                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Creative">
                <Name>Jerry</Name>
                <AgentStart>
                  <Placement x="0.5" y="227" z="0" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <AbsoluteMovementCommands/>
                  <ObservationFromNearbyEntities>
                    <Range name="NearbyEntities" xrange="200" yrange="200" zrange="200"/>
                  </ObservationFromNearbyEntities>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

agent_host = MalmoPython.AgentHost()

try:
    if "--debug" in sys.argv:
      sys.argv.remove("--debug")
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec(f"data-{time.time()}.tgz")
my_mission_record.recordObservations()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    my_mission.forceWorldReset()
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running...")

# Loop until mission ends:
while world_state.is_mission_running:
    input_text = input("Enter text: ")
    task = helpers.get_prediction(input_text)
    exec(f"helpers.task_{task}(agent_host)")
    world_state = agent_host.getWorldState()
    
    for error in world_state.errors:
        print("Error:", error.text)

print()
print("Mission ended")
# Mission has ended.
