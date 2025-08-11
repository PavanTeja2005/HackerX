import subprocess
import psutil
import sys
import time

def measure_memory_usage(script_path):
    # Run the script as a subprocess and monitor its memory usage
    process = subprocess.Popen([sys.executable, script_path])
    max_mem = 0
    try:
        while process.poll() is None:
            try:
                p = psutil.Process(process.pid)
                mem = p.memory_info().rss
                if mem > max_mem:
                    max_mem = mem
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
        print(f"Max Memory: {max_mem // 1024} KB")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    measure_memory_usage("app.py")