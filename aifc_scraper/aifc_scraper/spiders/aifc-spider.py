import scrapy
import pandas as pd


### Instructions: 
### Go to this dir: alfc/aifc_scraper/aifc_scraper
### Execute: scrapy crawl aifc-scraper -O "../../data/2024/scrapped_data.json"


# load idx and construct links
idx = pd.read_csv(r"..\..\data\2024\idx_to_scrape_value_and_wage.csv", sep=';')
list_idx = list(idx['player_id'].values)
for index in range(len(list_idx)):
    list_idx[index] = 'https://www.fifaindex.com/de/player/' + str(list_idx[index])
    print(list_idx[index])


class AIFCSpider(scrapy.Spider):
    name = "aifc-scraper"
    start_urls = list_idx #['https://www.fifaindex.com/de/player/120533']

    def parse(self, response):
        print(response.url)
        elements = response.css("p.data-currency.data-currency-euro>span::text").getall() 
        
        try:
            yield {
                'value_eur' : elements[0],
                'wage_eur' : elements[1]
            }
        except: # no data found
            yield {
                'value_eur' : 0,
                'wage_eur' : 0
            }