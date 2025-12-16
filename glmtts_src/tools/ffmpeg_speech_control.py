# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import subprocess
import shutil
import sys
import os
from pathlib import Path
from tqdm import tqdm

def check_ffmpeg_availability():
    """
    Checks if ffmpeg is installed and accessible in the system PATH.
    """
    if shutil.which('ffmpeg') is None:
        print("[ERROR] 'ffmpeg' is not found. Please install it and add it to your PATH.")
        sys.exit(1)

def change_audio_speed(input_path, output_path, speed_factor):
    """
    Changes the speed of an audio file using ffmpeg's 'atempo' filter.
    
    Args:
        input_path (Path): Path to the source audio.
        output_path (Path): Path where the modified audio will be saved.
        speed_factor (float): Speed multiplier (e.g., 1.2 for 20% faster).
    """
    # Validation: atempo filter usually supports 0.5 to 2.0. 
    # For values outside this, we would need to chain filters, 
    # but for typical speech control, this range is sufficient.
    if not (0.5 <= speed_factor <= 100.0):
        # Note: Modern ffmpeg handles >2.0, but it's good to be aware of limits.
        pass 

    command = [
        'ffmpeg',
        '-y',                 # Overwrite output file without asking
        '-i', str(input_path),
        '-filter:a', f'atempo={speed_factor}',
        '-vn',                # Disable video (audio only)
        '-loglevel', 'error', # Suppress verbose output
        str(output_path)
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to process {input_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch Audio Speed Control Tool using FFmpeg")
    
    # Required arguments
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input file path OR directory path containing .wav files.")
    
    # Optional arguments
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output directory. If not specified, creates a sibling directory.")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                        help="Speed factor (e.g., 1.2 for 1.2x speed). Default is 1.0.")
    
    args = parser.parse_args()

    # 1. System Check
    check_ffmpeg_availability()

    input_path = Path(args.input)
    
    # 2. Determine Output Directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Auto-generate output folder name if not provided
        suffix = f"_speed_{args.speed}x"
        if input_path.is_dir():
            output_dir = input_path.parent / (input_path.name + suffix)
        else:
            output_dir = input_path.parent / "processed_audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # 3. Collect Files
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # Recursively find wav files, or just top level. Adjust glob as needed.
        files_to_process = list(input_path.glob("*.wav"))
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        return

    if not files_to_process:
        print("[WARN] No .wav files found to process.")
        return

    print(f"[INFO] Processing {len(files_to_process)} files with speed factor {args.speed}...")

    # 4. Process Batch
    for src_file in tqdm(files_to_process, desc="Converting"):
        # Construct output filename, preserving name but adding suffix
        new_filename = f"{src_file.stem}_speed_{args.speed}{src_file.suffix}"
        dest_file = output_dir / new_filename
        
        change_audio_speed(src_file, dest_file, args.speed)

    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()