import asyncio 
import logging
import time
from pathlib import Path
import multiprocessing as mp

# Assuming these are your custom classes
from controller import DroneController
from vision import VisionService

# ---------------------------------------------------------
# PROCESS 1: DRONE FLIGHT CONTROLLER
# ---------------------------------------------------------
def move_drone(command_queue):
    """Runs in a background process, receiving commands via a Queue."""
    print("Process 1: Initializing Flight Controller...")
    controller = DroneController(logger=logging)
    controller.connect()
    
    while True:
        # This will pause and wait until the main process sends a number into the queue
        choice = command_queue.get() 

        if choice == 1:
            controller.takeoff()
        elif choice == 2:
            controller.move(0, 0, 5, 2)
        elif choice == 3:
            controller.move(0, 0, -5, 2)
        elif choice == 4:
            controller.move(0, -5, 0, 2)
        elif choice == 5:
            controller.move(0, 5, 0, 2)
        elif choice == 6:
            controller.disconnect()
            break # Exit the loop and end the process

# ---------------------------------------------------------
# PROCESS 2: CAMERA VISION (ASYNC)
# ---------------------------------------------------------
def vision_async_loop():
    """The actual async loop that handles the camera."""
    print("Process 2: Initializing Camera Vision...")
    project_root = Path(__file__).resolve().parent
    vision = VisionService(project_root, logger=logging)
    
    while True:
        # Because VisionService uses async, we MUST await them here
        frame = vision.get_frame()
        vision.save_frame(frame)
        
        # Use asyncio.sleep instead of time.sleep in an async function!
        time.sleep(2) 

def extract_frames():
    """Runs in a background process, kicking off the async event loop."""
    # This creates a fresh event loop specifically for this separate process
    asyncio.run(vision_async_loop())

# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
def main():
    print("Starting Drone System...")
    
    # 1. Create a communication queue
    command_queue = mp.Queue()
    
    # 2. Setup the processes (pass the queue to the drone process)
    process1 = mp.Process(target=move_drone, args=(command_queue,))
    process2 = mp.Process(target=extract_frames)

    process1.start()
    process2.start()

    # 3. Handle User Input in the MAIN process (where the terminal is)
    try:
        while True:
            print("\n1- takeoff | 2- up | 3- down | 4- left | 5- right | 6- disconnect")
            try:
                choice = int(input("Enter : "))
                
                # Send the user's choice through the pipe to the background process
                command_queue.put(choice)
                
                if choice == 6:
                    break # Stop asking for input if we disconnect
            except ValueError:
                print("Please enter a valid number.")
                
    except KeyboardInterrupt:
        # Safety catch: If you press Ctrl+C, tell the drone to disconnect safely
        print("\nEmergency Stop Triggered. Disconnecting...")
        command_queue.put(6)

    # Wait for the flight controller to finish its disconnect sequence
    process1.join()
    
    # Because process2 is an infinite while loop, we have to forcefully terminate it 
    # once the drone is disconnected.
    process2.terminate()
    process2.join()
    
    print("Main program finished. Drone safely shut down.")

if __name__ == "__main__":
    main()