import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_run():
    print("🚀 Sending Run Request...")
    payload = {
        "query": "RAG 시스템에서 검색 품질을 높이는 방법은?",
        "thread_id": "test_thread_1"
    }
    try:
        response = requests.post(f"{BASE_URL}/run", json=payload)
        response.raise_for_status()
        data = response.json()
        run_id = data["run_id"]
        print(f"✅ Run Initiated: {run_id}")
    except Exception as e:
        print(f"❌ Failed to start run: {e}")
        return

    # Poll status
    print("⏳ Polling status...")
    for _ in range(60): # Wait up to 60 seconds (might timeout if long)
        try:
            status_res = requests.get(f"{BASE_URL}/status/{run_id}")
            status_data = status_res.json()
            status = status_data["status"]
            print(f"   Status: {status}")
            
            if status == "completed":
                print("✅ Run Completed!")
                # Get result
                result_res = requests.get(f"{BASE_URL}/result/{run_id}")
                result = result_res.json()
                print("\n=== Final Result ===")
                print(str(result)[:500] + "...") # Print preview
                break
            elif status == "failed":
                print("❌ Run Failed!")
                break
            
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Error polling: {e}")
            time.sleep(2)

if __name__ == "__main__":
    test_run()
