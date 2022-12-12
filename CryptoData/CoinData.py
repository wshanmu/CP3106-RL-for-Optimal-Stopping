import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import datetime
import argparse


def get_data(date, item=100):
    start = datetime.datetime.now()
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    s = Service("./chromedriver.exe")  # change to the location of chromedriver
    driver = webdriver.Chrome(service=s, options=chrome_options)
    url = "https://coinmarketcap.com/historical/" + date + "/"
    driver.get(url)
    driver.maximize_window()

    # Scroll down mouse wheel
    pre_height = 0
    while True:
        driver.execute_script("scrollBy(0,1200)")
        time.sleep(0.5)
        now_height = driver.execute_script("return document.documentElement.scrollHeight;")
        if now_height == pre_height:
            break
        pre_height = now_height

    html = BeautifulSoup(driver.page_source, 'lxml')
    tables = html.find_all('table')[2]
    df = pd.read_html(str(tables))[0]
    df.drop(df.columns[10:1000], axis=1, inplace=True)
    if len(df) > item:
        df.drop(df[df.Rank > item].index, inplace=True)
    end = datetime.datetime.now()
    print("Successfully obtain Data for " + date)
    print('Time： {time}'.format(time=(end - start)))
    return df


def date_range(beginDate, endDate, interval=1):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(interval)
        date = dt.strftime("%Y%m%d")
    return dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", type=str, default="20220801", help="start date")
    parser.add_argument("--ed", type=str, default="20221110", help="end date")
    parser.add_argument("--item", type=int, default=100, help="how many items to be recorded")
    parser.add_argument("--save", type=str, default="./Data/", help="save folder")
    args = parser.parse_args()
    start_date = args.sd
    end_date = args.ed
    item = args.item
    save = args.save
    date_list = date_range(start_date, end_date)
    for date in date_list:
        df = get_data(date, item=item)
        df.to_csv(save + date + '.csv', index=False)
