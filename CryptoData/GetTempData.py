import pickle
import datetime
import pandas as pd
import argparse


def date_range(beginDate, endDate, interval=1):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(interval)
        date = dt.strftime("%Y%m%d")
    return dates


def get_item_index(index, date):
    market_list = []
    price_list = []
    name_list = []
    df = pd.read_csv("./Data/" + date + ".csv")
    index = min(index, df.index.size)
    for i in range(index):
        name_list.append(df.loc[df['Rank'] == i + 1, 'Name'].values)
        price_str = df.loc[df['Rank'] == i + 1, 'Price'].values
        if price_str.size > 0:
            price_list.append(float(price_str[0][1:].replace(',', '')))
        else:
            price_list.append(0)
        market_str = df.loc[df['Rank'] == i + 1, 'Market Cap'].values
        if market_str.size > 0:
            market_list.append(float(market_str[0][1:].replace(',', '')))
        else:
            market_list.append(0)
    return market_list, price_list, name_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", type=str, default="20130506", help="start date")
    parser.add_argument("--ed", type=str, default="20221129", help="end date")
    parser.add_argument("--index", type=int, default=20, help="how many items to be recorded")
    parser.add_argument("--save_name", type=str, default="Data", help="save name for the .pkl file")
    args = parser.parse_args()
    start_date = args.sd
    end_date = args.ed
    index = args.index
    save_name = args.save_name
    date_list = date_range(start_date, end_date, interval=1)
    Data = {}
    for date in date_list:
        market_list, price_list, name_list = get_item_index(index=index, date=date)
        for i in range(len(name_list)):
            Data[str(date) + str(name_list[i][0]) + '_price'] = price_list[i]
            Data[str(date) + str(name_list[i][0]) + '_market'] = market_list[i]
        continue

    with open('./Data/' + save_name + '.pkl', 'wb') as f:
        pickle.dump(Data, f)
    print("Successfully Generated the Temp Data File!")
