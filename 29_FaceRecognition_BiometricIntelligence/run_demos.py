#!/usr/bin/env python3
"""
Face Recognition & Biometric Intelligence - Demo Launcher
Session 29: Art of Programming

This launcher helps you easily run any of the 5 demo scripts.
"""

import os
import sys
import subprocess

def print_header():
    print("\n" + "="*70)
    print("SESSION 29: FACE RECOGNITION & BIOMETRIC INTELLIGENCE")
    print("="*70)
    print("\n🎓 Master the foundations of computer vision and biometric AI")

def print_menu():
    print("\n📋 Available Demos:")
    print("\n1. Part 1: Face Detection")
    print("   └─ Basic face detection with HOG detector")
    print("   └─ Snapshot demo: Capture and detect faces")

    print("\n2. Part 2: Facial Landmarks")
    print("   └─ 68-point facial landmark detection")
    print("   └─ Smile detection and face orientation analysis")

    print("\n3. Part 3: Face Recognition")
    print("   └─ 128D face encoding and comparison")
    print("   └─ Two-step demo: Register → Verify")

    print("\n4. Part 4: Real-time Face Tracking")
    print("   └─ Live face tracking with unique IDs")
    print("   └─ Posture monitoring & focus time tracking")

    print("\n5. Part 5: Expression Detection")
    print("   └─ Real-time emotion recognition")
    print("   └─ Head pose estimation & gesture control")

    print("\n0. Exit")
    print("\n" + "-"*70)

def run_script(script_name):
    """Run a Python script in the current directory"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"\n❌ Error: {script_name} not found!")
        return

    print(f"\n▶️  Running {script_name}...")
    print("-"*70 + "\n")

    try:
        # Run the script
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Script exited with error: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")

    print("\n" + "-"*70)
    input("\nPress ENTER to return to menu...")

def main():
    scripts = {
        '1': 'part1_finding_faces.py',
        '2': 'part2_understanding_facial_geometry.py',
        '3': 'part3_face_recognition.py',
        '4': 'part4_realtime_face_tracking.py',
        '5': 'part5_expression_detection.py'
    }

    while True:
        print_header()
        print_menu()

        choice = input("Select a demo (0-5): ").strip()

        if choice == '0':
            print("\n👋 Thanks for exploring Session 29!")
            print("Keep mastering the art of programming! 🚀\n")
            break

        elif choice in scripts:
            run_script(scripts[choice])

        else:
            print("\n❌ Invalid choice. Please select 0-5.")
            input("Press ENTER to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
