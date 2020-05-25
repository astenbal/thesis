from enum import Enum
import operator

# Allowed comparison functions in query
# IMPORTANT : Order matters, no function whose regex is a substring of another can precede the other
class ComparisonType(Enum):
    GTE = {'name': '>=' , 'regex' : '(?:>=)', 'func' : operator.ge}
    LTE = {'name': '<=' , 'regex' : '(?:<=)', 'func' : operator.le}
    NE = {'name': '!=' , 'regex' : '(?:!=)', 'func' : operator.ne}
    EQ = {'name': '=' , 'regex' : '=', 'func' : operator.eq}
    GT = {'name': '>' , 'regex' : '>', 'func' : operator.gt}
    LT = {'name': '<' , 'regex' : '<', 'func' : operator.lt}
    UNKNOWN = {'name': 'UNKNOWN' , 'regex' : '(?:UNKNOWN)', 'func' : lambda data, value : False}