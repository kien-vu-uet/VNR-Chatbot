import scrapy
from bs4 import BeautifulSoup
import os
from typing import Tuple

ATTM_PATH = '/workspace/nlplab/kienvt/KLTN/proc0-crawl/attachments/web-info'
if not os.path.exists(ATTM_PATH):
    os.mkdir(ATTM_PATH)

class FuncSpiderSpider(scrapy.Spider):
    name = "func_spider"
    allowed_domains = ["vr.org.vn", "192.168", "vrqc.vn", "app.vr.org.vn"]
    start_urls = ["http://www.vr.org.vn/Pages/sitemap.aspx"]
    ignore_urls = ['van-ban', 'quy-chuan-tieu-chuan', 'tin-tuc-su-kien']
    current_item_id = 99999
    urls = []
    error_urls = []
    
    def is_ignore_url(self, url:str) -> bool:
        for p in self.ignore_urls:
            if p in url:
                return True
        return False
            
    def get_text(self, html:str) -> Tuple[str, str]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for data in soup(['style', 'script']):
                data.decompose()
            data = list(soup.stripped_strings)
            title = data[0].upper()
            return title, '\n'.join(data[1:]).replace('\u200b', '').strip()
        except:
            return None, None
    
        # text =  soup.text.strip('\n')
        # while text.find('<script') != -1:
        #     s = text.find('<script')
        #     e = text.find('/script>')
        #     assert e != -1, f'Got unexpected markdown structure!'
        #     text = text.replace(text[s:e+8], '')
        # while text.find('\n\n') != -1:
        #     text = text.replace('\n\n', '\n')
        # return text.strip('\n')

    def parse(self, response):
        left_side = response.css('div.full-left')
        right_side = response.css('div.full-right')

        self.urls += list(left_side.css('::attr(href)').getall()) + \
                     list(right_side.css('div.box-listnew')[1].css('::attr(href)').getall())
        # print(self.urls)
        # for url in urls:
        while len(self.urls) > 0:
            url_ = self.urls.pop(0)
            if self.is_ignore_url(url_):
                self.error_urls.append(url_)
                continue
            else:
                if url_.startswith('/'):
                    url_ = 'http://www.vr.org.vn' + url_
                elif url_.startswith('?'):
                    url_ = response.url + url_
                elif url_.startswith('#') or url_.startswith('javascript'):
                    continue
                if not url_.startswith('http'):
                    url_ = 'http://' + url_
                yield response.follow(url_, callback=self.parse_page)
                
        # next_page_url = response.css("div.paging a.current + a::attr(href)").get()
        # if next_page_url is not None:
        #     next_page_url = 'http://www.vr.org.vn/' + next_page_url.strip()
        #     yield response.follow(next_page_url, callback = self.parse)
        print(self.error_urls)
                
    def parse_page(self, response):
        title, para = self.get_text(response.css('div.full-left').get())
        if title is not None and para is not None:
            self.current_item_id += 1
            with open(os.path.join(ATTM_PATH, f'{self.current_item_id}-{title}.txt'), 'w', encoding='utf-8') as f:
                f.write(para)
                f.close()
            yield {
                "item_id": self.current_item_id,
                "page_url": response.url,
                "title": title,
            }
        else:
            self.error_urls.append(response.url)
        yield response.follow(response.url, callback=self.parse)
        


