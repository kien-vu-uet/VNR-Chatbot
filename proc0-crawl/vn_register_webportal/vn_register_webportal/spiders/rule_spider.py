import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
import re

class RuleSpiderSpider(scrapy.Spider):
    name = "rule_spider"
    allowed_domains = ["www.vr.org.vn"]
    start_urls = ["http://www.vr.org.vn/quy-chuan-tieu-chuan/Pages/default.aspx"]
    
    rules = [
        Rule(LinkExtractor(allow=r'\/default\.aspx(?:\?Page=\d)?'), callback='parse'),
        Rule(LinkExtractor(allow=r'\?ItemID=\d+'), callback='parse_rule_page'),
    ]

    def parse(self, response):
        table = response.css('div.project-type-item .tableList tr')
        for row in table:
            rule_url = row.css('::attr(href)').get()
            if rule_url is not None:
                rule_url = "http://www.vr.org.vn/quy-chuan-tieu-chuan/Pages/default.aspx" + rule_url.strip()
                yield response.follow(rule_url, callback = self.parse_rule_page)
            
            
        next_page_url = response.css("div.paging a.current + a::attr(href)").get()
        if next_page_url is not None:
            next_page_url = 'http://www.vr.org.vn/' + next_page_url.strip()
            yield response.follow(next_page_url, callback = self.parse)


    def parse_rule_page(self, response):
        rule_info = {}
        
        item_id = re.search(r'ItemID=\d+', response.url).group(0)
        try:
            item_id = int(item_id.lstrip("ItemID="))
        except: pass
        rule_info['item_id'] = item_id
        rule_info['page_url'] = response.url.strip()
        
        try: rule_info['title'] = response.css('div.qc-qp-tc h3.title::text').get().strip()
        except: rule_info['title'] = None
        
        try: rule_info['symbol_number'] = response.css('div.qc-qp-tc h4.code::text').get().strip()
        except: rule_info['symbol_number'] = None
        
        try: rule_info['field'] = response.css('div.qc-qp-tc div.cate::text')[0].get().strip()
        except: rule_info['field'] = None
        
        try: rule_info['rule_category'] = response.css('div.qc-qp-tc div.cate::text')[1].get().strip()
        except: rule_info['rule_category'] = None
        
        try: rule_info['description'] = response.css('div.qc-qp-tc div.des::text').get().strip()
        except: rule_info['description'] = None

        
        attachment_url = response.css('div.qc-qp-tc ::attr(href)').get()
        if attachment_url is not None:
            if not attachment_url.strip().startswith('http'):
                attachment_url = 'http://www.vr.org.vn/' + attachment_url.strip()
            rule_info['attachment_url'] = attachment_url
            
        doc_url = response.css('div.qc-qp-tc div.doc ::attr(data)').get()
        try:
            obj_url = re.match(r'^.*(?=\?)', doc_url.strip()).group(0)
            rule_info['attachment_url'] = obj_url
        except: pass
        
        yield rule_info
        
        
        