# test_modal_basic.py
import modal

app = modal.App("test-basic")

@app.function()
def hello():
    return "Modal is working!"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Result: {result}")