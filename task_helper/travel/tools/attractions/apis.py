import pandas as pd
from pandas import DataFrame
from pathlib import Path
from task_helper.travel.utils.func import extract_before_parenthesis
from task_helper.travel.utils.paths import travel_dataset_root


def _default_attractions_path() -> Path:
    return travel_dataset_root() / "database" / "attractions" / "attractions.csv"


class Attractions:
    def __init__(self, path=None):
        self.path = Path(path) if path else _default_attractions_path()
        self.data = pd.read_csv(self.path).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results
