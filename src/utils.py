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