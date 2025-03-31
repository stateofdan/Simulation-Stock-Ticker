import json
import os

class Config():
    __internal_vars = None

    __schema = {
        "config_dir": str,
        "roles_file": str,
        "users_file": str,
        "group_role": str,
        "test":int,
        "stock_data": {
            "template":{
                "symbol": str,
            }
        },
    }
    # Cannot contain sets
    '''__schema = {
        "name": str,
        "version": str,
        "settings": {
            "option1": bool,
            "option2": int,
            "nested": {
                "sub_option1": [int],
                "sub_option2": dict
            }
        },
        "items": [str],
        "set" :[int]
    }'''

    def __init__(self, file_path):
        self.file_path = file_path
        self.__internal_vars = self.load_config()

    def load_config(self):
        try:
            print (f'cwd={os.getcwd()}')
            with open(self.file_path, 'r') as file:
                config = json.load(file)
                self.validate_config(config, self.__schema)
                print (f'loaded config from:{self.file_path}')
                return config
        except FileNotFoundError as e:
            print(f"Error: The file {self.file_path} was not found.")
            raise e
        except json.JSONDecodeError as e:
            print(f"Error: The file {self.file_path} is not a valid JSON.")
            raise e
        except Exception as e:
            raise e

    def validate_config(self, config: any, schema: any, path: str=""):
        for key, value_type in schema.items():
            print  (f'testing {path + key}-> {value_type}')
            if key not in config:
                raise ValueError(f"Missing key: {path + key}")
            if isinstance(value_type, dict):
                if not isinstance(config[key], dict):
                    raise TypeError(f"Expected dict at {path + key}, got {type(config[key]).__name__}")
                self.validate_config(config[key], value_type, path + key + ".")
            elif isinstance(value_type, list):
                if not isinstance(config[key], list):
                    raise TypeError(f"Expected list at {path + key}, got {type(config[key]).__name__}")
                for item in config[key]:
                    if not isinstance(item, value_type[0]):
                        raise TypeError(f"Expected {value_type[0].__name__} in list at {path + key}, got {type(item).__name__}")
            elif not isinstance(config[key], value_type):
                raise TypeError(f"Expected {value_type.__name__} at {path + key}, got {type(config[key]).__name__}")

    # ----- Static Methods -----
    @staticmethod
    def __get_all_item_types(current_node: dict=None, parent_key: str="", include_target: bool = True, get_type: bool = False) -> list:
        if current_node is None:
            raise ValueError(f'current_node cannot be none')
        if not isinstance(current_node, dict):
            raise ValueError(f'current_node is not a dictionary {type(current_node)}')
        keys = []
        
        for key, value in current_node.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if not include_target:
                keys.append(full_key)
            elif get_type:
                keys.append({full_key: type(value)})
            else:
                data = value
                if isinstance(value, dict):
                    data = type(value)        
                keys.append({full_key: data})
            if isinstance(value, dict):
                keys.extend(Config.__get_all_item_types(value, full_key, include_target, get_type))
        return keys

    # ----- Private Methods -----

    # ----- Schema Methods -----
    @classmethod
    def get_config_schema_all_keys(cls) -> list[str]:
        return Config.__get_all_item_types(cls.__schema, include_target=False)

    @classmethod
    def get_config_schema_structure(cls) ->list[dict]:
        return Config.__get_all_item_types(cls.__schema, get_type = False)


    # ----- Data Methods -----
    def get(self, key, default=None):
        keys = [k.strip() for k in key.split(".")]
        value = self.__internal_vars
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
        
    def get_keys(self, current_dict: dict=None, parent_key: str="") -> list[str]:
        if current_dict is None:
            current_dict = self.__internal_vars
        keys = []
        for key, value in current_dict.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            keys.append(full_key)
            if isinstance(value, dict):
                keys.extend(self.get_keys(value, full_key))
        return keys
    
    def print_config_keys(self):
        keys = self.get_keys()
        for idx, key in enumerate(keys, start=1):
            print (f'{idx}: {key}')

    