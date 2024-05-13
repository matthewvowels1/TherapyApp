import psutil
import time

def get_streamlit_process():
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if 'streamlit' in proc.info['name']:
            return proc
    return None

def monitor_streamlit(duration=60, interval=1):
    streamlit_process = get_streamlit_process()
    if not streamlit_process:
        print("Streamlit process not found.")
        return

    max_cpu = 0
    max_memory = 0

    for _ in range(int(duration / interval)):
        cpu = streamlit_process.cpu_percent()
        memory = streamlit_process.memory_percent()
        max_cpu = max(max_cpu, cpu)
        max_memory = max(max_memory, memory)
        time.sleep(interval)

    return max_cpu, max_memory

if __name__ == "__main__":
    duration = 60  # Monitor for 60 seconds
    interval = 1   # Collect data every 1 second
    max_cpu, max_memory = monitor_streamlit(duration, interval)
    print(f"Max CPU Usage: {max_cpu}%")
    print(f"Max Memory Usage: {max_memory}%")
    print(psutil.cpu_count(), psutil.virtual_memory().total)