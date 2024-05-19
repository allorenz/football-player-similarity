library(httr)
library(rvest)
response <- GET("https://fbref.com/en/players/8652a85c/Lois-Openda")
html = read_html(response)

# Accesses the whole table
# html %>% html_elements(xpath='//*[@id="stats_standard_dom_lg"]') %>% html_table(header=TRUE)

# Acceses in a smaller view 
# html %>% html_elements(xpath='//*[@id="stats_standard_dom_lg"]/tbody') %>% html_table(header=TRUE)

url = "https://suchen.mobile.de/fahrzeuge/search.html?dam=false&isSearchRequest=true&ms=20100%3B17%3B%3B&ref=quickSearch&s=Car&sb=rel&vc=Car"
response <- GET(url, user_agent("Hey!"))
html = read_html(response)

html %>% html_elements(xpath='//*[@id="root"]/div/div[7]/div[2]/article[1]/section[1]/div/h1')
