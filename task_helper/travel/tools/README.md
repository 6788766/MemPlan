### Resource Fetching Tools (tools/<domain>/apis.py)

  1. Flights (tools/flights/apis.py:1-38)
      - Input: origin city, destination city, departure date (strings).
      - Output: Pandas DataFrame of matching flights (or a string if none).
      - Usage: ReactAgent calls Flights.run() whenever the model issues FlightSearch[...]. The planner’s hard-constraint checker also inspects flight.data directly
        (evaluation/hard_constraint.py:1-150).
  2. Accommodations (tools/accommodations/apis.py:7-30)
      - Input: city name.
      - Output: DataFrame of accommodations with price, room type, house rules, occupancy; “no attraction” string if empty.
      - Usage: Triggered by AccommodationSearch[City] actions; evaluation checks the same tables for rule compliance.
  3. Restaurants (tools/restaurants/apis.py:6-50), Attractions (tools/attractions/apis.py:7-34)
      - Input: city name.
      - Output: DataFrame of venues; strings if not found.
      - Usage: Provide the meal/attraction slots; evaluation ensures diversity and correct cities.
  4. GoogleDistanceMatrix (tools/googleDistanceMatrix/apis.py:13-123)
      - Input: origin, destination, mode (‘driving’ or ‘taxi’).
      - Output: Formatted string or dict with duration, distance, cost (run_for_evaluation returns a dict).
      - Usage: Called for GoogleDistanceMatrix[...] actions and for transport cost estimation in both the ReAct environment and budget calculators.
  5. Cities (tools/cities/apis.py:1-21)
      - Input: state name.
      - Output: Python list of cities in that state (or ValueError).
      - Usage: When the LLM issues CitySearch[State], it receives the canonical list, enabling state-level trips.

———

### Notebook Storage (tools/notebook/apis.py:1-23)

  - Role: Acts as scratch memory for tool outputs.
  - Input: DataFrame plus a short description.
  - Output: Confirmation string; list_all() returns every entry as {index, Short Description, Content} with DataFrames converted to strings.
  - Usage: The agent writes after each successful tool call so the planner sees consistent evidence.

———

### Planner Runtime (tools/planner/…)

  1. tool/planner/apis.py:50-199
      - Classes: Planner (direct or CoT), ReactPlanner, ReactReflectPlanner.
      - Inputs: Planner.run(text, query) takes a stringified notebook plus the user query. ReAct variants alternate Thought/Action/Observation steps internally.
      - Outputs: Plan string (or 'Max Token Length Exceeded.'). ReAct versions also return a scratchpad for logging.
  2. Environment (tools/planner/env.py:1-126)
      - Validates a single-day subplan by cross-referencing flight, restaurant, accommodation, and distance data, summing costs for the CostEnquiry tool.
      - Input: dict representing a day’s plan (including people count).
      - Output: cost string or error messages.
  3. Sole Planning Entry (tools/planner/sole_planning.py:63-113)
      - Iterates over dataset splits, feeding reference_information and query to a selected planner strategy.
      - Inputs: set type, model name, strategy, output dir (argparse).

  - Outputs: writes generated_plan_{n}.json with planner responses.

  ———

