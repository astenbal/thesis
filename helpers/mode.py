from enum import Enum

# Enum to select the type of query being done, includes the name to look for and the function to execute for the query
class Mode(Enum):
    UNKNOWN = {'name': 0, 'func' : 0}
    AVG = {'name': 'AVG', 'func' : lambda d: d.mean()}
    SUM = {'name': 'SUM', 'func' : lambda d: d.sum()}