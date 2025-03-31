from matplotlib import pyplot as plt
import matplotlib

from flask import Flask, render_template
import requests
import base64
import json
import io
from matplotlib.figure import Figure
from datetime import datetime, timezone, timedelta   

import scipy.stats as stats
import numpy as np
import pandas as pd
from pandas_datareader import data as wb

import yfinance as yf
import seaborn as sns
import StockStats as ss

# Global Variables

''' TODO
 - add a way to render the previous days trading data
 - Update the x axis to be related to times.
 - Add the logos to the web page
 - provide a secure log in for all users and teams
 - check that the long run creates the same data points. - Done

'''

trading_dates = {'start': '2025-03-17', 'end': '2025-03-21'}
trading_times = {'start': '08:00:00', 'end': '16:30:00'}



app = Flask(__name__)

api_key = "JO1FX7Y5NL3ORKJE"

day_start = None

stock_model = {}

def get_stock_price_stats(symbol):
    print (f"\nGetting stock data for {symbol}")
    stock_model['symbol'] = symbol
    #This returns a dataframe
    df_stock_data = yf.download(tickers=symbol, period='5d', interval='1m')
    print (df_stock_data.head())
    # This returns a series from the prior dataframe
    # A series is more like a dictionary
    ds_volume_data = df_stock_data['Volume'][symbol]
    print (ds_volume_data.index.name)
    print (ds_volume_data.head())

    results = ss.fit_distribution(ds_volume_data)
    print (f'{results['best_fit_aic']}')
    print (f'{results['best_fit_bic']}')
    best_fit = results['results'][results['best_fit_aic']]
    
    stock_model['volume'] = {'dist': results['best_fit_aic'], 'params': best_fit['params']}
    print (stock_model['volume'])

    ds_close_data = df_stock_data['Close'][symbol]
    results = ss.fit_distribution(ds_close_data)
    print (f'{results['best_fit_aic']}')
    print (f'{results['best_fit_bic']}')
    best_fit = results['results'][results['best_fit_aic']]
    stock_model['close'] = {'dist': results['best_fit_aic'], 'params': best_fit['params']}
    print (stock_model['close'])


    log_return = np.log(1+ds_close_data.pct_change())
    u = log_return.mean()
    var = log_return.var()
    stddev = log_return.std()
    drift = u - (0.5 * var)
    stock_model['log_pct_drift'] = {'drift': drift, 'mean': u, 'var': var, 'stddev': stddev}
    print (f'Returns:')
    print (f"Drift: {drift}, Mean: {u}, Variance: {var}, Std Dev: {stddev}")



    sns.displot(ds_volume_data, kde=True, log_scale=True) 
    plt.title(f'{symbol} Volume trades')
    plt.xlabel("daily return")
    plt.ylabel("Volume")
    sns.displot(ds_close_data, kde=True, log_scale=True) 
    plt.title(f'{symbol} close price')
    plt.xlabel("daily return")
    plt.ylabel("price")
    sns.displot(log_return.iloc[1:], kde=True, log_scale=True) 
    plt.title(f'{symbol} PCT returns')
    plt.xlabel("daily return")
    plt.ylabel("Volume")
    plt.show()
    print ("\n\n")

def get_stock_stats(symbol):
    symbol = 'GOOG'  # Replace with your desired stock symbol
    start_date = '2000-01-01'

    # Fetch historical data
    stock_data = yf.download(symbol, start=start_date)
    print ()
    df_close = stock_data['Close']
    #df_close.plot(figsize=(10, 6), title=f'{symbol} Close Price', ylabel='Price')
    print (df_close.head())
    log_return = np.log(1+df_close.pct_change())
    #plt.figure(figsize=(10, 6))
    #sns.displot(log_return.iloc[1:])
    #plt.xlabel("daily return")
    #plt.ylabel("frequency")
    #plt.show()

    u = log_return.mean()[symbol]
    var = log_return.var()[symbol]
    stddev = log_return.std()[symbol]
    print (f'Returns:')
    print (f"Mean: {u}, Variance: {var}, Std Dev: {stddev}")
    drift = u - (0.5 * var)
    print (f"Drift: {drift}, u: {u}, var: {var}")

    df_volume = stock_data['Volume']
    print (df_volume.describe())
    log_volume = np.log(1+df_volume.pct_change())
    #plt.figure(figsize=(10, 6))
    #sns.displot(log_volume.iloc[1:])
    #plt.xlabel("daily return")
    #plt.ylabel("Volume")
    #plt.show()
    u = log_volume.mean()[symbol]
    var = log_volume.var()[symbol]
    stddev = log_volume.std()[symbol]
    drift = u - (0.5 * var)
    print (f'Volume:')
    print (f"Drift: {drift}, Mean: {u}, Variance: {var}, Std Dev: {stddev}")




def old():
    function = 'TIME_SERIES_DAILY_ADJUSTED'
    outputsize = 'full'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&outputsize={outputsize}'

    response = requests.get(url)
    data = response.json()
    print (data)
    # Extract the time series data
    time_series = data['Time Series (Daily)']

    # Convert the time series data to a DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Filter data since 2000-01-01
    df = df[df.index >= '2000-01-01']

    # Select only the adjusted close column
    df = df[['5. adjusted close']]
    df.plot(figsize=(10, 6), title=f'{symbol} Adjusted Close Price', ylabel='Price')


def get_stock_price_by_1minute(symbol):
    #This gets the last 30days of stock prices for the given symbol
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    time_series = data["Time Series (1min)"]
    
    stock_prices = {}
    for time, price in time_series.items():
        stock_prices[time] = price["4. close"]
    return stock_prices

stock_prices = {"A": 100}

stock_data = {"A": {'last_update': None, "prices": [0], "stock_adapt":[2], "stock_volitility":[1], "rng": np.random.default_rng(42)}}

#Mean: 0.0008122482683790715, Variance: 0.000368114767680968, Std Dev: 0.01918631719952967
stock_data2 = {"A": {'last_update': None, "prices": {}, "stock_adapt":[0.001], "stock_volitility":[0.02], "rng": np.random.default_rng(42)}}
end_trading = False
initial_stock = {'open':100, 'high':100, 'low':100, 'close':100, 'vol':0}


def update_stock_price(group):
    working_data = stock_data[group]
    working_data["prices"].insert(0, working_data["prices"][0] + working_data["stock_adapt"][0] * working_data["rng"].normal(0, working_data["stock_volitility"][0]))


def stock_analysis(group_id):
    '''
    This is used to do some stock analysis on a current groups stock price to 
    make sure it is in line with the current settings for the group

    '''
    df_stock = pd.DataFrame.from_dict(stock_data2[group_id]['prices'], orient='index')
    df_close = df_stock['close']
    log_return = np.log(1+df_close.pct_change())
    #print (log_return)
    u = log_return.mean()
    var = log_return.var()
    stddev = log_return.std()
    drift = u - (0.5 * var)
    print (f"Group {group_id} Current: Drift: {drift}, Std Dev: {stddev}, Mean: {u}, Variance: {var}")
    print (f"Group {group_id} Target:  Drift: {stock_data2[group_id]['stock_adapt'][0]}, Std Dev: {stock_data2[group_id]['stock_volitility'][0]}")
    

def update_stock_price2(group):
    now = datetime.now(timezone.utc)
    now_time = now.time()
    start_time = datetime.strptime(trading_times['start'], '%H:%M:%S').time()
    end_time = datetime.strptime(trading_times['end'], '%H:%M:%S').time()
    '''if now_time < start_time or now_time > end_time:
        if not end_trading:
            end_trading = True
            print ("Trading has ended")
        return'''
    
    if stock_data2[group]['last_update'] is None:
        today = datetime.today().date()
        initial_time = datetime.combine(today, datetime.strptime(trading_times['start'], '%H:%M:%S').time())
        stock_data2[group]['last_update'] = initial_time.strftime('%Y-%m-%d %H:%M:%S')
        print (f'Setting new Initial Time: {stock_data2[group]["last_update"]}')
        stock_data2[group]['prices'][stock_data2[group]['last_update']] = initial_stock

    last_update = datetime.strptime(stock_data2[group]['last_update'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    
    if now - last_update < timedelta(seconds=30):
        print (f"Not enough time has passed {(now-last_update).total_seconds()}")
        return

    updates_needed = int((now - last_update).total_seconds() // 30)
    print (f"Updating {updates_needed} times")
    working_data = stock_data2[group]['prices']

    current_update = last_update
    prev_ohlcv_key = current_update.strftime('%Y-%m-%d %H:%M:%S')
    prev_ohlcv = working_data[prev_ohlcv_key]
    
    for _ in range (updates_needed):
        if False:
            if global_rng.uniform(0, 1) >0.75:
                update_stock_adapt_relative(group, global_rng.uniform(-1, 1))
                print (f"Adapted stock price  {stock_data[group]['stock_adapt'][1]} -> {stock_data[group]['stock_adapt'][0]}")
            if global_rng.uniform(0, 1) >0.90:
                vol_update = global_rng.uniform(low=-1, high=1)
                update_stock_volitility_relative(group, vol_update)
                print (f"Adapted stock volitility  {vol_update}: {stock_data[group]['stock_volitility'][1]} -> {stock_data[group]['stock_volitility'][0]}")
        current_update = current_update + timedelta(seconds=30)
        current_ohlcv_key = current_update.strftime('%Y-%m-%d %H:%M:%S')
        open = prev_ohlcv['close']
        #close = open + stock_data2[group]['stock_adapt'][0] * stock_data2[group]['rng'].normal(0, stock_data2[group]['stock_volitility'][0])
        close = open * np.exp(stock_data2[group]['stock_adapt'][0] + stock_data2[group]['stock_volitility'][0] * stock_data2[group]['rng'].normal(0, 1))
        high = max(open, close) + stock_data2[group]['rng'].uniform(0, 1)
        low = min(open, close) - stock_data2[group]['rng'].uniform(0, 1)
        dist = getattr(stats, stock_model['volume']['dist'])
        vol = int(dist.rvs(*stock_model['volume']['params']))
        print (f'trading volume: {vol}')
        working_data[current_ohlcv_key] = {'open':open, 'high':high, 'low':low, 'close':close, 'vol':0}
        prev_ohlcv_key = current_ohlcv_key

    stock_data2[group]['last_update'] = current_update.strftime('%Y-%m-%d %H:%M:%S')

    print (len(stock_data2[group]['prices']))
    #print (stock_data2[group]['prices'])
    #stock_value = working_data["prices"][0] + working_data["stock_adapt"][0] * working_data["rng"].normal(0, working_data["stock_volitility"][0])

def update_stock_adapt_absolute(group, value):
    stock_data[group]["stock_adapt"].insert(0, value)

def update_stock_adapt_relative(group, value):
    stock_data[group]["stock_adapt"].insert(0, stock_data[group]["stock_adapt"][0] + value)

def update_stock_volitility_absolute(group, value):
    value = max(value, 0.1)
    stock_data[group]["stock_volitility"].insert(0, value)

def update_stock_volitility_relative(group, value):
    value = max(stock_data[group]["stock_volitility"][0] + value, 0.1)
    stock_data[group]["stock_volitility"].insert(0, value)


def accelerate_trading():
    #need to move everything over to the OHLV format and have timestamps for entries
    pass

def test():
    rng = np.random.default_rng(42)
    for i in range(12*60):
        update_stock_price("A")
        if rng.uniform(0, 1) >0.75:
            update_stock_adapt_relative("A", rng.uniform(-1, 1))
            print (f"Adapted stock price  {stock_data['A']['stock_adapt'][1]} -> {stock_data['A']['stock_adapt'][0]}")
        if rng.uniform(0, 1) >0.90:
            vol_update = rng.uniform(low=-1, high=1)
            update_stock_volitility_relative("A", vol_update)
            print (f"Adapted stock volitility  {vol_update}: {stock_data['A']['stock_volitility'][1]} -> {stock_data['A']['stock_volitility'][0]}")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(stock_data["A"]["prices"])
    time_steps = np.arange(len(stock_data["A"]["prices"]))
    trend_line = np.polyfit(time_steps, stock_data["A"]["prices"], 1)
    trend_line_values = np.polyval(trend_line, time_steps)
    plt.plot(trend_line_values, label="Trend Line", linestyle='--')
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("Stock Price")
    plt.show()


@app.route('/')
def hello_world():
    return 'Hello, World!'





group_stock_deltas = {
    "A": {"adapt": [], "volitility": []},
}

def generate_stock_delta_plot(group_id):
    fig = plt.figure(figsize=(10, 6))
    group_stock_deltas[group_id]["adapt"].append(stock_data2[group_id]["stock_adapt"][0])
    group_stock_deltas[group_id]["volitility"].append(stock_data2[group_id]["stock_volitility"][0])
    stock_adapt_set = group_stock_deltas[group_id]["adapt"]
    stock_volitility_set = group_stock_deltas[group_id]["volitility"]
    time_steps = np.arange(0, len(stock_adapt_set) * 30, 30)
    # Plot stock_adapt as a line plot
    plt.plot(time_steps, stock_adapt_set, label='Stock Adapt', color='blue')

    # Plot stock_volitility as an area around the stock_adapt points
    plt.fill_between(time_steps, np.array(stock_adapt_set) - np.array(stock_volitility_set), np.array(stock_adapt_set) + np.array(stock_volitility_set), color='blue', alpha=0.2)

    # Adding labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Stock Adapt')
    plt.title('Stock Adapt and Volatility Over Time')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return plot_base64

def generate_plot(group_id):


    data_set = list(stock_data[group_id]["prices"])[::-1]

    change_val = stock_data[group_id]["prices"][0] - stock_data[group_id]["prices"][-1]
    colour = (19/255, 115/255, 51/255) if change_val > 0 else (165/255, 14/255, 14/255)

    time_steps = np.arange(len(data_set))
    trend_line = np.polyfit(time_steps, data_set, 3)
    time_steps = np.arange(len(data_set) + min(int(len(data_set)*0.15), 30))
    trend_values = np.polyval(trend_line, time_steps)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(data_set, color= colour, label="Stock Price")
    y_min = np.min(data_set)
    times = np.arange(len(data_set))
    plt.fill_between(times, data_set, y_min, interpolate=True, color=colour, alpha=0.3)
    
    plt.plot(trend_values, label="Stock Trend", linestyle='--')

    # Plot a horizontal line at y=0.5
    plt.axhline(y=data_set[0], color='#191919', linestyle='--', linewidth=2)    
    plt.text(time_steps[-1], data_set[0], f'Prev Close\n GBX {data_set[0]}', color='#191919', fontsize=10, verticalalignment='bottom', horizontalalignment='right')



    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title(f'{company_name} - Group {group_id} Stock Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return plot_base64

def generate_stock_plot(group_id):

    stock_ohlcv = stock_data2[group_id]['prices']
    data_set = [v['close'] for v in stock_ohlcv.values()]

    #data_set = list(stock_data[group_id]["prices"])[::-1]

    change_val = data_set[-1] - data_set[0]
    colour = (19/255, 115/255, 51/255) if change_val > 0 else (165/255, 14/255, 14/255)

    print (data_set[0], data_set[-1])
    time_steps = np.arange(len(data_set))
    trend_line = np.polyfit(time_steps, data_set, 1)
    time_steps = np.arange(len(data_set) + min(int(len(data_set)*0.15), 30))
    trend_values = np.polyval(trend_line, time_steps)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(data_set, color= colour, label="Stock Price")
    y_min = np.min(data_set)
    times = np.arange(len(data_set))
    plt.fill_between(times, data_set, y_min, interpolate=True, color=colour, alpha=0.3)
    
    plt.plot(trend_values, label="Stock Trend", linestyle='--')

    # Plot a horizontal line at y=0.5
    plt.axhline(y=data_set[0], color='#191919', linestyle='--', linewidth=2)    
    plt.text(time_steps[-1], data_set[0], f'Prev Close\n GBX {data_set[0]}', color='#191919', fontsize=10, verticalalignment='bottom', horizontalalignment='right')



    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title(f'{company_name} - Group {group_id} Stock Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return plot_base64

company_name = "Oaklead Plc"
symbol = "OAK"

@app.route('/group/<group_id>')
def group(group_id):
    global day_start
    if day_start is None:
        today = datetime.today().date()
        day_start = datetime.combine(today, datetime.strptime(trading_times['start'], '%H:%M:%S').time()).replace(tzinfo=timezone.utc)
        print (f'Setting new Initial Time: {day_start}')
        

    #update_stock_price(group_id)
    update_stock_price2(group_id)
    stock_analysis(group_id)
    '''if global_rng.uniform(0, 1) >0.75:
        update_stock_adapt_relative(group_id, global_rng.uniform(-1, 1))
        print (f"Adapted stock price  {stock_data[group_id]['stock_adapt'][1]} -> {stock_data[group_id]['stock_adapt'][0]}")
    if global_rng.uniform(0, 1) >0.90:
        vol_update = global_rng.uniform(low=-1, high=1)
        update_stock_volitility_relative(group_id, vol_update)
        print (f"Adapted stock volitility  {vol_update}: {stock_data[group_id]['stock_volitility'][1]} -> {stock_data[group_id]['stock_volitility'][0]}")
    '''
    latest_update_key = stock_data2[group_id]['last_update']
    day_start_key = day_start.strftime('%Y-%m-%d %H:%M:%S')
    #change_val = stock_data[group_id]["prices"][0] - stock_data[group_id]["prices"][-1]
    
    day_open_value = stock_data2[group_id]['prices'][day_start_key]['open']
    latest_close_value = stock_data2[group_id]['prices'][latest_update_key]['close']
    change_val = latest_close_value - day_open_value
    
    print (day_open_value, latest_close_value,change_val)
    change_dir = "up" if change_val > 0 else "down"

    change_val = np.round(np.abs(change_val), 2)
    change_percent = (change_val / day_open_value)* 100
    now = datetime.now(timezone.utc)
    formatted_date_time = now.strftime('%b %d, %I:%M:%S %p UTC')
    data = {"company_name": company_name, "group_id": group_id, "symbol": symbol, 
            "share_price": np.round(latest_close_value,2),
            "change_val": change_val,
            "change_dir": change_dir,
            "change_percent": np.round(change_percent, 2),
            "date_time": formatted_date_time,
            "plot_base64":generate_stock_plot(group_id),
            "adapt_plot_base64":generate_stock_delta_plot(group_id)}

    #plot_base64 = generate_plot(group_id)
    return render_template('group.html', data = data)

if __name__ == '__main__':
    #get_stock_stats('GOOG')
    get_stock_price_stats('BYRN')
    if True:
        matplotlib.use('Agg')
        app.debug = True
        #test()
        app.run()