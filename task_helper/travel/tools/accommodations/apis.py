import pandas as pd
from pandas import DataFrame
from pathlib import Path
from task_helper.travel.utils.func import extract_before_parenthesis
from task_helper.travel.utils.paths import travel_dataset_root


def _default_accommodations_path() -> Path:
    return travel_dataset_root() / "database" / "accommodations" / "clean_accommodations_2022.csv"


class Accommodations:
    def __init__(self, path=None):
        self.path = Path(path) if path else _default_accommodations_path()
        self.data = pd.read_csv(self.path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no attraction in this city."
        
        return results
    
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        return results
