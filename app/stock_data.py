from wonderwords import RandomWord
import os
import json



class StockData:
    __internal_vars = {}
    def __init__(self, group_data: list[str]):
        r = RandomWord()
        self.__internal_vars['data'] = {}
        print (f'loading group data for {group_data}')
        for key in group_data:
            self.__internal_vars['data'][key] = {'test':' '.join(r.random_words(3, word_min_length=5, word_max_length=10)).strip().capitalize()}

    @staticmethod
    def load_users(file: str, role_type: str, ignore:bool = True) -> list[str]:
        required_keys = {'username', 'role', 'password'}
        return_list = []
        if not os.path.exists(file):
            raise FileNotFoundError(f"The file '{file}' does not exist.")

        with open(file, 'r') as f:
            data = json.load(f)

        if not data:
            raise ValueError("The file is empty or does not contain any data.")

        if not isinstance(data, list):
            raise ValueError("The file format is invalid. Expected a list at the root.")
        
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                if not ignore:
                    raise ValueError(f'User Entry at {idx} is not a valid dictionary: type {type(entry)}')
                print (f'User Entry at {idx} is not a valid dictionary')
            if not set(entry.keys()) == required_keys:
                if not ignore:
                    raise ValueError(f'User Entry at {idx} invalid dictionary keys: {entry.keys()}->{required_keys}')
                print (f'User Entry at {idx} invalid dictionary keys: {entry.keys()}->{required_keys}')
            for key, item in entry.items():
                if not isinstance(item, str):
                    if not ignore:
                        raise ValueError(f'User Entry at {idx} invalid data type for key {key}->{type(item)}')
                    print (f'User Entry at {idx} invalid data type for key {key}->{type(item)}')
            if entry['role'] == role_type:
                return_list.append(entry['username'])

        return return_list
