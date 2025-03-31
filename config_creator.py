from helpers.config_loader import Config
import json

def create_example_dict(schema: dict, parent_key: str = "") -> dict:
    """Recursively create an example dictionary from the schema."""
    example_dict = {}
    for key, value_type in schema.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value_type, dict):
            # If the value is a dictionary, recurse
            example_dict[key] = create_example_dict(value_type, full_key)
        else:
            # Set a placeholder value based on the type
            if value_type == str:
                example_dict[key] = f"example_{key}"
            elif value_type == int:
                example_dict[key] = 0
            elif value_type == bool:
                example_dict[key] = False
            elif value_type == list:
                example_dict[key] = []
            else:
                example_dict[key] = None
    return example_dict


def prompt_user_for_values(schema_keys: list[dict], example_dict: dict):
    """Prompt the user for values for each key in the schema."""
    for idx, entry in enumerate(schema_keys, start=1):
        key, val_type = next(iter(entry.items())) 
        keys = key.split(".")
        current_dict = example_dict
        for k in keys[:-1]:
            current_dict = current_dict[k]
        last_key = keys[-1]

        # Skip if the item is a dictionary
        if isinstance(current_dict[last_key], dict):
            continue

        # Prompt the user for input
        while True:
            value = input(f"{idx}: Enter value for {key} (current: {current_dict[last_key]}): ").strip()
            if value == "":
                print('Value cannot be empty')
            else:
                try:
                    value = val_type(value)
                    break
                except ValueError:
                    print (f'config value must convert to {val_type.__name__}')

        if value:
            # Convert the input to the appropriate type
            if isinstance(current_dict[last_key], int):
                current_dict[last_key] = int(value)
            elif isinstance(current_dict[last_key], bool):
                current_dict[last_key] = value.lower() in ["true", "yes", "1"]
            elif isinstance(current_dict[last_key], list):
                current_dict[last_key] = value.split(",")  # Assume comma-separated input for lists
            else:
                current_dict[last_key] = value


def amend_values(schema_keys: list[str], example_dict: dict):
    """Allow the user to amend values by selecting an index."""
    while True:
        print("\nCurrent configuration:")
        for idx, key in enumerate(schema_keys, start=1):
            keys = key.split(".")
            current_dict = example_dict
            for k in keys[:-1]:
                current_dict = current_dict[k]
            last_key = keys[-1]
            print(f"{idx}: {key} -> {current_dict[last_key]}")

        choice = input("\nEnter the index of the key to amend (or 'done' to finish): ").strip()
        if choice.lower() == "done":
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(schema_keys):
                key = schema_keys[idx]
                keys = key.split(".")
                current_dict = example_dict
                for k in keys[:-1]:
                    current_dict = current_dict[k]
                last_key = keys[-1]

                # Prompt for a new value
                while True:
                    value = input(f"Enter new value for {key} (current: {current_dict[last_key]}): ").strip()
                    if value == "":
                        print('Value cannot be empty')
                    else:
                        break
                if value:
                    if isinstance(current_dict[last_key], int):
                        current_dict[last_key] = int(value)
                    elif isinstance(current_dict[last_key], bool):
                        current_dict[last_key] = value.lower() in ["true", "yes", "1"]
                    elif isinstance(current_dict[last_key], list):
                        current_dict[last_key] = value.split(",")
                    else:
                        current_dict[last_key] = value
            else:
                print("Invalid index. Please try again.")
        else:
            print("Invalid input. Please enter a valid index or 'done'.")

def main():
    schema_keys = Config.get_config_schema_all_keys()
    for idx, key in enumerate(schema_keys, start=1):
        print (f'{idx}: {key}')
    schema_keys_and_types = Config.get_config_schema_structure()
    for idx, item in enumerate(schema_keys_and_types, start=1):
        key, value = next(iter(item.items()))
        print (f'{idx}: {key} -> {value}')# Create an example dictionary from the schema
    example_dict = create_example_dict(Config._Config__schema)

    # Prompt the user for values
    print("Enter values for the configuration:")
    prompt_user_for_values(schema_keys_and_types, example_dict)

    # Show the completed configuration and allow amendments
    print("\nCompleted configuration:")
    amend_values(schema_keys, example_dict)

    # Write the configuration to a JSON file
    output_file = "generated_config.json"
    with open(output_file, "w") as f:
        json.dump(example_dict, f, indent=2)
    print(f"\nConfiguration saved to {output_file}")

    # Load the configuration back into a Config object to validate
    print("\nValidating the configuration...")
    config = Config(output_file)
    print("Configuration loaded successfully!")

if __name__ == "__main__":
    main()