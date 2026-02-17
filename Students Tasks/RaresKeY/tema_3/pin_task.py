import time
from IPython.display import clear_output
from tqdm import trange

PIN_CORRECT = 1234

tries_left = 3
sleep_time = 0
pin_ok = False
RUN = True # CHANGE TO TRUE TO RUN


def countdown(sec, msg="Try again in"):
    frag = 10
    for s in trange(sec * frag, leave=False, desc="Waiting",
                    bar_format="Waiting:|{bar}| {n:.1f}/{total}s...",
                    unit='s',
                    unit_scale=0.1):
        # print(f"\r{msg} {s / frag}s...", end="", flush=True)
        time.sleep(1 / frag)
    # print("\r" + " " * 40 + "\r", end="")  # clear line


def loading_bar():
    print()

if RUN == True:
    while not pin_ok:
        print("Introduce PIN in input field")
        cand_pin = input("PIN: ")

        while not cand_pin.isdigit():
            print("We only accept digits (0123456789)")
            cand_pin = input("PIN (0123456789): ")

        clear_output(wait=True)  # clears this cell’s output

        cand_pin = int(cand_pin)

        if cand_pin == PIN_CORRECT:
            pin_ok = True
            break

        if tries_left == 1:
            print("Exceeded retries attempts. Account was locked down (Contact support for help unlocking)")
            break

        wait = 15 if tries_left == 3 else 60
        countdown(wait)
        tries_left -= 1
else:
    print("skipped")

if pin_ok:
    print("Access Granted")
else:
    print("Access Denied")