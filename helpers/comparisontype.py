from enum import Enum

# Allowed comparison functions in query
# IMPORTANT : Order matters, no function whose regex is a substring of another can precede the other
class ComparisonType(Enum):
    GTE = {'name': '>=' , 'regex' : '(?:>=)', 'func' : lambda data, value: data >= value}
    LTE = {'name': '<=' , 'regex' : '(?:<=)', 'func' : lambda data, value: data <= value}
    NE = {'name': '!=' , 'regex' : '(?:!=)', 'func' : lambda data, value: data != value}
    E = {'name': '=' , 'regex' : '=', 'func' : lambda data, value : data == value}
    GT = {'name': '>' , 'regex' : '>', 'func' : lambda data, value: data > value}
    LT = {'name': '<' , 'regex' : '<', 'func' : lambda data, value: data < value}
    UNKNOWN = {'name': 'UNKNOWN' , 'regex' : '(?:UNKNOWN)', 'func' : lambda data, value : False}