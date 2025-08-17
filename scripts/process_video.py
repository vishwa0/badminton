import argparse
from main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run badminton court monitoring')
    parser.add_argument('--video', required=True, help='Path to input video file')
    args = parser.parse_args()
    main(args.video)