# modal_test.py
import modal

modal.App= modal.App("lerobot-test")

@modal.App.function()
def say_hi():
    print("Hello from Modal!")
