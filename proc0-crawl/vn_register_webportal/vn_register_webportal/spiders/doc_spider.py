import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
import re

class DocSpiderSpider(scrapy.Spider):
    name = "doc_spider"
    allowed_domains = ["www.vr.org.vn"]
    start_urls = ["http://www.vr.org.vn/van-ban/Pages/home.aspx"]
    
    rules = [
        Rule(LinkExtractor(allow=r'\/home\.aspx(\?Page\=\d)?'), callback='parse'),
        Rule(LinkExtractor(allow=r'\?ItemID=\d'), callback='parse_doc_page'),
    ]

    def parse(self, response):
        table = response.css('div.boxfull-leftright.leftbox-home table')
        for row in table.css('tr td'):
            doc_url = row.css('a::attr(href)').get()
            if doc_url is not None:
                doc_url = 'http://www.vr.org.vn/' + doc_url.strip()                
                yield response.follow(doc_url, callback = self.parse_doc_page)
                
        next_page_url = response.css("div.paging a.current + a::attr(href)").get()
        if next_page_url is not None:
            next_page_url = 'http://www.vr.org.vn/' + next_page_url.strip()
            yield response.follow(next_page_url, callback = self.parse)

    
    def parse_doc_page(self, response):
        doc_info = {}
        
        item_id = re.search(r'ItemID=\d+', response.url).group(0)
        try:
            item_id = int(item_id.lstrip("ItemID="))
        except: pass
        
        doc_info['item_id'] = item_id
        doc_info['page_url'] = response.url.strip()
        
        content_1 = response.css('div#content_1')
        content_2 = response.css('div#content_2')
        # content_3 = response.css('div#content_3')
        try:
            doc_info['effectiveness']   = content_1.css('div.vbInfo ul li::text')[0].get().strip()
        except: 
            doc_info['effectiveness']   = None
        try:
            doc_info['date_of_issue']   = content_1.css('div.vbInfo ul li::text')[1].get().strip()
        except:
            doc_info['date_of_issue']   = None
        try:
            doc_info['effective_date']  = content_1.css('div.vbInfo ul li::text')[2].get().strip()
        except:
            doc_info['effective_date']  = None
        
        url = content_1.css("div.toanvan a::attr(href)").get()
        if url is not None and url.strip().startswith('/'):
            url = self.allowed_domains[0] + url
            doc_info['attachment_url'] = url
            
        
        obj = response.css('div.toanvan object::attr(data)').get()
        try:
            obj_url = re.match(r'^.*(?=\?)', obj.strip()).group(0)
            doc_info['attachment_url'] = obj_url
        except: pass 
        
        download_url = response.css('div#divShowDialogDownload a::attr(href)').get()
        if download_url is not None:
            doc_info['attachment_url'] = download_url.strip()
        
        title = content_2.css('div.vbProperties tr td.title').css('::text').get().strip()
        doc_info['title'] = title
        
        meta_data = {}
        
        table = content_2.css('tr td.tdborder')
        key = None
        value = None
        for i in range(len(table)):
            if table[i].attrib['class'].startswith('label'):
                key = table[i].css('b::text').get().strip()
            else:
                value = table[i].css('::text').get().strip()
            
                if key is not None:
                    meta_data[key] = value
        
        doc_info['meta_data'] = meta_data
        
        yield doc_info
