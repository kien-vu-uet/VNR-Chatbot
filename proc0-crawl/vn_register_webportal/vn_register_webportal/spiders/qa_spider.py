import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
import re

class QaSpiderSpider(scrapy.Spider):
    name = "qa_spider"
    allowed_domains = ["thuvienphapluat.vn"]
    start_urls = [
        # "https://thuvienphapluat.vn/hoi-dap-phap-luat/tim-tu-van?searchType=1&q=&searchField=21&page=1",
        "https://thuvienphapluat.vn/hoi-dap-phap-luat/giao-thong-van-tai"
    ]
    
    rules = [
        Rule(LinkExtractor(allow=r'\/hoi-dap-phap-luat\/tim-tu-van.+page=\d+'), callback='parse'),
        Rule(LinkExtractor(allow=r'\/hoi-dap-phap-luat\/giao-thong-van-tai'), callback='parse'),
        Rule(LinkExtractor(allow=r'\w+.\.html'), callback='parse_qa_page_2'),
    ]

    def parse(self, response):
        section = response.css('div.col-md-9 section article')
        for article in section:
            article_url = article.css('a::attr(href)').get()
            if article_url is not None:          
                yield response.follow(article_url, callback = self.parse_qa_page_2)
                
        next_page_urls = response.css('ul.pagination li.page-item a.page-link ::attr(href)')
        for url in next_page_urls:
            url = url.get()
            if url is not None:
                yield response.follow(url, callback = self.parse)

    
    def parse_qa_page(self, response):
        article = response.css('article div.col-md-9')
        edited_q = article.css('h1::text').get().strip()
        origin_q = article.css('section.news-content strong::text').get().strip()
        answers  = [p.get().strip() for p in article.css('section.news-content p::text')]
        answer   = ' '.join(answers)
        
        yield {
            "query": origin_q,
            "context": answer,
            "url": response.url
        }
        
        yield {
            "query": edited_q,
            "context": answer,
            "url": response.url
        }

    def parse_qa_page_2(self, response):
        article = response.css('article div.col-md-9 section.news-content')
        contents = [c.strip() for c in article.css('::text').extract() if c.strip().__len__() > 0]
        accordions = article.css('div.accordion-body li strong ::text').getall()
        q_pos = [contents.index(accor.strip()) for accor in accordions]
        q_pos = [contents.index(accor.strip(), i+1) for accor, i in zip(accordions, q_pos)]
        q_pos.append(-1)
        for i in range(len(accordions)):
            question = accordions[i]
            yield {
                "query": question,
                # "content": contents, 
                "context": ' '.join(contents[q_pos[i]+1:q_pos[i+1]]),
                "url": response.url,
                # "qpos": q_pos
            }
        
        
    

