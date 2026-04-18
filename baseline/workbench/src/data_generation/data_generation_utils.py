import pandas as pd

# WorkBench baseline expects a pandas Timestamp here for prompt injection.
# Keep the canonical WorkBench time (midnight) for determinism.
HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T00:00:00")

