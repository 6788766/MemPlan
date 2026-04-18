from pathlib import Path
from pandas import DataFrame
from task_helper.travel.utils.paths import travel_dataset_root


def _default_cities_path() -> Path:
    return travel_dataset_root() / "database" / "background" / "citySet_with_states.txt"


class Cities:
    def __init__(self ,path=None) -> None:
        self.path = Path(path) if path else _default_cities_path()
        self.load_data()
        print("Cities loaded.")

    def load_data(self):
        cityStateMapping = open(self.path, "r").read().strip().split("\n")
        self.data = {}
        for unit in cityStateMapping:
            city, state = unit.split("\t")
            if state not in self.data:
                self.data[state] = [city]
            else:
                self.data[state].append(city)
    
    def run(self, state) -> dict:
        if state not in self.data:
            return ValueError("Invalid State")
        else:
            return self.data[state]
