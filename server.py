import socket
import time
import os
import subprocess
import threading

log_base = "./server_logs/"
os.makedirs(log_base, exist_ok=True)

def get_server_status(port):
    '''
    Return the status of the server on the given port.

    Args:
        port (int): The port number to check.

    Returns:
        str:    "running" if the server is running,
                "loading" if the server is still loading,
                "error" if the server failed to load.
    '''
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', port))
            return "running"
    except ConnectionRefusedError as e:
        return "loading"
    except Exception as e:
        return "error"

def wait_for_server(port, timeout=100, interval=5):
    '''
    Wait for the server to load on the given port.

    Args:
        port (int): The port number to check.
        timeout (int): The maximum time to wait for the server to load.
        interval (int): The time to wait between checks.
    
    Returns:
        bool:   True if the server is running,
                False if the server failed to load.
    '''
    start = time.time()
    while time.time() - start < timeout:
        status = get_server_status(port)
        if status == "running":
            print(f"Server is running on port {port}.")
            return True
        elif status == "loading":
            print(f"Server is loading on port {port}.")
            time.sleep(interval)
        else:
            print(f"Server failed to load on port {port}.")
            return False
    print(f"Timeout waiting for server to load on port {port}.")
    return False

def start_server(command, port, timeout=100):
    '''
    Start the server using the given command, wait for it to load, and return the server process.

    Args:
        command (str): The command to start the server.
        port (int): The port number to check.
        timeout (int): The maximum time to wait for the server to load.

    Returns:
        subprocess.Popen: The server process if the server is running,
        None: None if the server failed to load.
    '''
    log_file = f"{log_base}servers_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    command = f"{command} > {log_file} 2>&1"
    server = threading.Thread(target=subprocess.run, args=(command,), kwargs={'shell': True}, daemon=True)
    server.start()
    if wait_for_server(port, timeout):
        return server
    else:
        return None