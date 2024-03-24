import re


# add mio and k to values 
def adjust_money_appearance(x):
    x = int(x)
    if(x >= 1000000):
        x = x / 1000000
        x = "€" + str(x) + " Mio"
        return x
    if(x < 1000000 and x >= 10000):
        x = x / 1000
        x = "€" + str(x) + "k"
        return x
    return "€" + str(x)


# Define a regular expression pattern
pattern = r'\s([A-Z]+)$'

# Function to clean the column
def clean_nation(nation):
    match = re.search(pattern, str(nation))
    if match:
        return match.group(1)
    else:
        return nation