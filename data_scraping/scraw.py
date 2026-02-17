import random
import time
from asyncio import timeout
import json
from idlelib.rpc import response_queue
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import codecs
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from fake_useragent import UserAgent

class TikiDataset:
    def __init__(self):
        self.session = requests.Session()
        # configure retry to prevent connection loss
        retry_strategy = Retry(
            total=5,
            backoff_factor = 1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=50,
            pool_maxsize=50
        )

        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
            'Referer': 'https://tiki.vn/',
            "Connection": "keep-alive",
        }

        # random user agent
        self.ua = UserAgent()

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.max_products_per_category = 100
        self.max_reviews_per_product = 1000
        self.base_url = "https://tiki.vn"
        self.data = []
        self.session.headers.update(self.headers)

    def get_categories(self):
        '''
            fetch categories from tiki
        '''
        # main categories list
        url = 'https://api.tiki.vn/raiden/v2/menu-config?platform=desktop&30122025=1'
        # add a delay before sending request
        time.sleep(random.uniform(1,3))
        try:
            headers = self.headers.copy()
            headers['User-Agent'] = self.ua.random
            response = self.session.get(url , timeout=15, headers=headers)
            response.raise_for_status()
            data = response.json()
            main_categories = []
            items = data['menu_block']['items']
            for item in items:
                category = item['text']
                category_url = item['link']
                category_id = category_url.split('/c')[-1]
                if category_id != '': category_id = int(category_id)
                urlKey = category_url.split('/c')[0].split('/')[-1]
                main_categories.append({
                    'category_id': category_id,
                    'category': category,
                    'urlKey': urlKey,
                    'category_url': category_url
                })
            return main_categories[1:]
        except requests.exceptions.ConnectionError as e:
            time.sleep(10)
            return []
        except Exception as e:
            return []

    # for each category, retrieve product infomation
    def get_products_per_category(self, category):
        product_list = []
        page = 1
        while len(product_list) < self.max_products_per_category:
            url = (f"https://tiki.vn/api/personalish/v1/blocks/listings?limit=40&category="
                   f"{category['category_id']}&page={page}&urlKey={category['urlKey']}")
            # add a delay before sending requests
            time.sleep(random.uniform(3,6))
            try:
                headers = self.headers.copy()
                headers['User-Agent'] = self.ua.random
                response = self.session.get(url, timeout=30, headers=headers)
                if response.status_code != 200: break
                data = response.json()
                products = data['data']
                if not products: break
                for product in products:
                    id = product['id']
                    name = product['name']
                    url_key = product['url_key']
                    url_path = product['url_path']
                    review_count = product['review_count']
                    product_list.append({
                        'product_id': id,
                        'product_name':name,
                        'url_key':url_key,
                        'url_path': url_path,
                        'review_count': review_count
                    })
                page += 1
                time.sleep(random.uniform(2,4))
            except requests.exceptions.ConnectionError as e:
                time.sleep(15)
                continue
            except Exception as e:
                break
        return product_list

    def get_reviews_per_product(self, product):
        reviews = []
        page = 1

        while True:
            if len(reviews) >= self.max_reviews_per_product: break
            url = f"https://tiki.vn/api/v2/reviews?limit=20&product_id={product['product_id']}&page={page}"
            time.sleep(random.uniform(3,5))
            try:
                headers = self.headers.copy()
                headers['User-Agent'] = self.ua.random
                response = self.session.get(url,headers=headers,timeout=30)
                if response.status_code != 200:break
                data = response.json().get('data', [])
                if not data: break
                for review in data:
                    id = review['id']
                    title = review['title']
                    content = review['content']
                    rating = review['rating']
                    reviews.append({
                        'review_id':id,
                        'label': title,
                        'review': content,
                        'rating': rating
                    })
                paging = response.json().get("paging", {})
                if page >= paging.get("last_page", page):
                    break
                page += 1
                time.sleep(random.uniform(2,3))
            except requests.exceptions.ConnectionError as e:
                time.sleep(20)
                continue
            except Exception as e: break
        return reviews
    def crawl_data(self):
        categories = self.get_categories()
        if not categories: return
        for category in categories:
            print(f"Processing the category: {category['category']}")
            products = self.get_products_per_category(category)
            for product in products:
                print(f"Processing the product: {product['product_name']}")
                reviews = self.get_reviews_per_product(product)
                for review in reviews:
                    if not review['review']: continue
                    self.data.append({
                        'category': category['category_id'],
                        'category_name': category['category'],
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'review_id': review['review_id'],
                        'review': review['review'],
                        'rating': review['rating'],
                        'label': review['label']
                    })

                if len(self.data) % 1000 == 0:
                    self.save_data(f"tiki_reviews_temp_{len(self.data)}.csv")
                time.sleep(random.uniform(5,8))
            time.sleep(random.uniform(10, 15))
        self.save_data("tiki_dataset.csv")
    def save_data(self, filename:str):
        try:
            df = pd.DataFrame(self.data)
            df.to_csv(fr"C:\Users\NinhDB\PycharmProjects\tiki-sentiment-analytics\data_scraping\{filename}", index=False,
                      encoding="utf-8-sig")
        except Exception as e:
            print('File save error!')
def main():
    tiki = TikiDataset()
    tiki.crawl_data()
    # data = tiki.get_categories()
    # print(data)
if __name__=="__main__": main()