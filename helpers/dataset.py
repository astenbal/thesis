from enum import Enum

# Enum with all datasets and their names, used to select proper ctgan model and sql table
class Dataset(Enum):
    DIAB = 'diabetes'
    KIDNEY = 'kidney'
    BREAST = 'breast'
    UNKNOWN = '?'
