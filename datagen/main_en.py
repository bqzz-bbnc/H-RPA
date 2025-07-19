#!/usr/bin/env python3
"""
English Version Data Generator Main Entry
For generating English role-playing datasets
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from generator import DataGenerator


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="English Version Role-Playing Data Generator")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--world", "-w", required=True, help="World name")
    parser.add_argument("--role", "-r", required=True, help="Role name")
    parser.add_argument("--api-key", "-k", help="OpenAI API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create data generator
    generator = DataGenerator(
        world=args.world,
        role=args.role,
        config_path=args.config
    )
    
    # If API key is provided via command line, override config file setting
    if args.api_key:
        # Directly modify config dictionary
        keys = 'openai.api_key'.split('.')
        config_dict = generator.config._config
        for key in keys[:-1]:
            config_dict = config_dict[key]
        config_dict[keys[-1]] = args.api_key
    
    # Set language to English
    generator.config._config['language'] = 'en'
    
    # Execute data generation process
    try:
        print(f"Starting to generate English dataset for {args.role} from {args.world}")
        
        # Execute complete generation process
        generator.run()
        
        print(f"Data generation completed!")
        print(f"Output directory: {generator.config.get('paths.all_dir')}")
            
    except KeyboardInterrupt:
        print("User interrupted the data generation process")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred during data generation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        # Print complete error stack trace
        import traceback
        print("Complete error stack trace:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 