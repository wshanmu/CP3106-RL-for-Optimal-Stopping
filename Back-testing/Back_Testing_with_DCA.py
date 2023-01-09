import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


def date_range(beginDate, endDate, interval=1):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(interval)
        date = dt.strftime("%Y%m%d")
    return dates


def dateAdd(date, interval=1):
    dt = datetime.datetime.strptime(date, "%Y%m%d")
    dt = dt + datetime.timedelta(interval)
    date1 = dt.strftime("%Y%m%d")
    return date1


def get_price(content, date, name):
    if date+name+'_price' in content.keys():
        price = content[date + name + '_price']
    else:
        price = 0
    return price


def get_item_index(content, index, date):
    data_keys = list(content.keys())
    for i in range(len(data_keys)):
        if data_keys[i][0:8] == date:
            useful_keys = data_keys[i: i+(index * 2)]
            break
    name_list = [i[8:-6] for i in useful_keys[::2]]
    market_list = [content[i] for i in useful_keys[1::2]]
    price_list = [content[i] for i in useful_keys[::2]]
    return market_list, price_list, name_list


def get_name_index(content, index, date):
    data_keys = list(content.keys())
    for i in range(len(data_keys)):
        if data_keys[i][0:8] == date:
            useful_keys = data_keys[i: i + (index * 2)]
            break
    name_list = [i[8:-6] for i in useful_keys[::2]]
    return name_list


def get_market_and_price_index(content, date, name_list):
    data_keys = list(content.keys())
    useful_keys = []
    for i in range(len(data_keys)):
        if data_keys[i][0:8] == date:
            useful_keys.append(data_keys[i])
            if data_keys[i+1][0:8] != date:
                break
    market_list = []
    price_list = []
    for name in name_list:
        if date+name+'_market' in content.keys():
            market_list.append(content[date+name+'_market'])
            price_list.append(content[date+name+'_price'])
        else:
            market_list.append(0)
            price_list.append(0)
    return market_list, price_list


def AtomStrategy(content, invest=100, BeginDate='20190603', year=3, name='BTCBitcoin', interval=30):
    """ Buy single crypto-currency.
    Args:
        :param content: Temp Data of Crypto
        :param invest: Invest amount of each time
        :param BeginDate: Begin at
        :param year: Last for how many years
        :param interval: Investment cycle
        :return: Return / Investment Ratio
    """
    dt = datetime.datetime.strptime(BeginDate, "%Y%m%d")
    dt = dt + datetime.timedelta(weeks=52 * year)
    EndDate = dt.strftime("%Y%m%d")

    date_list = date_range(BeginDate, EndDate, interval=interval)
    total_USD = 0
    Amount = 0
    for date in date_list:
        inv = invest
        price = get_price(content, date, name)
        if price == 0:
            return 0, 0
        total_USD += inv
        Amount = Amount + inv / price

    # Calculate the average price of BTC around the EndDate
    price_avg = 0
    dt = datetime.datetime.strptime(date_list[-1], "%Y%m%d")
    dt1 = dt + datetime.timedelta(-6)
    date = dt1.strftime("%Y%m%d")
    date_list_2 = date_range(date, date_list[-1], interval=1)
    for date in date_list_2:
        prc = get_price(content, date, name)
        price_avg += prc
    price_avg = price_avg / 7

    current_asset = Amount * price_avg
    GLR = 100 * current_asset / total_USD  # Percentage

    return Amount, GLR


def IndexStrategy(content, invest=100, BeginDate='20190603', year=3, index=10, interval=30, MaxRatio=0.2):
    """ Index-10 && MaxRatio Strategy
    Args:
        :param content: Temp Data of Crypto
        :param invest: Invest amount of each time
        :param BeginDate: Begin at
        :param year: Last for
        :param index: Buy Top index assets
        :param interval: Investment cycle
        :param MaxRatio: set to 1 is Index-10
        :return: Return / Investment Ratio
    """
    dt = datetime.datetime.strptime(BeginDate, "%Y%m%d")
    dt = dt + datetime.timedelta(weeks=52 * year)
    EndDate = dt.strftime("%Y%m%d")

    date_list = date_range(BeginDate, EndDate, interval=interval)
    total_USD = 0
    Amount = {}
    season = 1
    name_list = get_name_index(content, index=index, date=date_list[0])
    for date_index in range(len(date_list)):
        date = date_list[date_index]
        if date_index * interval > season * 90:  # new season
            season += 1
            old_name_list = name_list
            name_list = get_name_index(content, index=index, date=date)
            new_name = []
            lost_name = []
            # Re-balancing every season:
            for name in name_list:
                if name not in old_name_list:
                    new_name.append(name)
            for name in old_name_list:
                if name not in name_list:
                    lost_name.append(name)
            if len(new_name) != 0:
                # sell old cryptos
                asset = 0
                for name in lost_name:
                    prc = get_price(content, date, name)
                    asset += prc * Amount[name]
                    Amount[name] = 0
                # buy new rising cryptos
                market_list, price_list = get_market_and_price_index(content, date, new_name)
                invest_in_new = [asset * i / sum(market_list) for i in market_list]
                amount_in_new = [i / j for i, j in zip(invest_in_new, price_list)]
                for i in range(len(new_name)):
                    name = new_name[i]
                    Amount[name] = amount_in_new[i]
                # print('rebalance' + str(old_name_list) + 'with' + str(new_name))

        market_list, price_list = get_market_and_price_index(content, date=date, name_list=name_list)
        total_USD += invest
        max_invest = invest * MaxRatio  # the max investment for each single coin
        invest_per_day = [invest * i / sum(market_list) for i in market_list]
        for i in range(index):
            if invest_per_day[i] <= max_invest:
                break
            else:
                rearrange_invest = invest_per_day[i] - max_invest
                invest_per_day[i] = max_invest
                temp_invest = [rearrange_invest * j / sum(market_list[i + 1:]) for j in market_list[i + 1:]]
                invest_per_day[i + 1:] = (np.array(invest_per_day[i + 1:]) + np.array(temp_invest)).tolist()

        amount_per_day = []
        for i, j in zip(invest_per_day, price_list):
            if j == 0:
                amount_per_day.append(0)
            else:
                amount_per_day.append(i/j)

        for i in range(index):
            if name_list[i] in Amount:
                Amount[name_list[i]] = Amount[name_list[i]] + amount_per_day[i]
            else:
                Amount[name_list[i]] = amount_per_day[i]

    # Calculate the current property in USD
    current_asset = 0
    for name in Amount:
        if Amount[name] != 0:
            prc = get_price(content, date_list[-1], name)
            current_asset += prc * Amount[name]
    GLR = 100 * current_asset / total_USD  # Percentage
    return GLR


def ShowEffectiveness(BeginDate='20160428', interval=30, year=5):
    dt = datetime.datetime.strptime('20220810', "%Y%m%d")
    dt = dt - datetime.timedelta(weeks=52 * year)
    EndDate = dt.strftime("%Y%m%d")
    F = open(r'./Data/Data.pkl', 'rb')
    content = pickle.load(F)
    date_list = date_range(BeginDate, EndDate, interval=1)
    GLR_list = [[], [], [], []]
    # BTC
    for date in date_list:
        Amount, GLR_B = AtomStrategy(content, invest=100, BeginDate=date, year=year, interval=interval, name='BTCBitcoin')
        GLR_list[0].append(GLR_B)
    print('Done: BTC')

    # ETH
    for date in date_list:
        Amount, GLR_E = AtomStrategy(content, invest=100, BeginDate=date, year=year, interval=interval, name='ETHEthereum')
        GLR_list[1].append(GLR_E)
    print('Done: ETH')

    # Index-10
    for date in date_list:
        GLR_index = IndexStrategy(content, invest=100, BeginDate=date, year=year, interval=interval, index=10, MaxRatio=1)
        GLR_list[2].append(GLR_index)
    print('Done: Index-10')

    # Max-Ratio
    for date in date_list:
        GLR_ratio = IndexStrategy(content, invest=100, BeginDate=date, year=year, interval=interval, index=10, MaxRatio=0.2)
        GLR_list[3].append(GLR_ratio)
    print('Done: Max-Ratio')

    # save the results for visualization
    save_name = ['BTC', 'ETH', 'Index10', 'MaxRatio']
    for i in range(4):
        a = np.array(GLR_list[2])
        np.save('GLR_year_%d_int_%d_%s.npy' % (year, interval, save_name[i]), a)


if __name__ == "__main__":
    for year in range(3):
        year = year + 1
        print('Begin Back Testing for %d-year investment' % year)
        ShowEffectiveness(BeginDate='20180812', year=year, interval=30)
        print('Done year:', year)

