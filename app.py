from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from binance.client import Client
from pycoingecko import CoinGeckoAPI
from models.laplace_predictor import LaplacePrediction
from models.ml_predictor import MLPrediction
from models.combined_predictor import CombinedPrediction
import traceback
import time
import requests

app = Flask(__name__)
CORS(app)

# Initialize clients
client = Client(None, None)

# Initialize CoinGecko client
cg = CoinGeckoAPI()

# Add CoinGecko ID mapping
COINGECKO_ID_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin',
    'SOL': 'solana',
    'ADA': 'cardano',
    'XRP': 'ripple',
    'DOGE': 'dogecoin',
    'DOT': 'polkadot',
    'MATIC': 'matic-network',
    'AVAX': 'avalanche-2',
    'LINK': 'chainlink',
    'UNI': 'uniswap',
    'ATOM': 'cosmos',
    'LTC': 'litecoin',
    'ETC': 'ethereum-classic',
    'XLM': 'stellar',
    'VET': 'vechain',
    'AAVE': 'aave',
    'SAND': 'the-sandbox',
    'MANA': 'decentraland',
    'APE': 'apecoin',
    'APT': 'aptos',
    'GALA': 'gala',
    'LDO': 'lido-dao',
    'CRV': 'curve-dao-token',
    'GMX': 'gmx',
    'IMX': 'immutable-x',
    'COMP': 'compound-governance-token',
    'OP': 'optimism',
    'ARB': 'arbitrum',
    'SUI': 'sui',
    'NEAR': 'near',
    'FTM': 'fantom'
}

@app.route('/api/coins')
def get_coins():
    # List of verified trading pairs on Binance
    verified_coins = [
        {'symbol': 'BTCUSDT', 'name': 'Bitcoin', 'shortName': 'BTC'},
        {'symbol': 'ETHUSDT', 'name': 'Ethereum', 'shortName': 'ETH'},
        {'symbol': 'BNBUSDT', 'name': 'Binance Coin', 'shortName': 'BNB'},
        {'symbol': 'SOLUSDT', 'name': 'Solana', 'shortName': 'SOL'},
        {'symbol': 'XRPUSDT', 'name': 'Ripple', 'shortName': 'XRP'},
        {'symbol': 'ADAUSDT', 'name': 'Cardano', 'shortName': 'ADA'},
        {'symbol': 'DOGEUSDT', 'name': 'Dogecoin', 'shortName': 'DOGE'},
        {'symbol': 'DOTUSDT', 'name': 'Polkadot', 'shortName': 'DOT'},
        {'symbol': 'MATICUSDT', 'name': 'Polygon', 'shortName': 'MATIC'},
        {'symbol': 'AVAXUSDT', 'name': 'Avalanche', 'shortName': 'AVAX'},
        {'symbol': 'LINKUSDT', 'name': 'Chainlink', 'shortName': 'LINK'},
        {'symbol': 'UNIUSDT', 'name': 'Uniswap', 'shortName': 'UNI'},
        {'symbol': 'ATOMUSDT', 'name': 'Cosmos', 'shortName': 'ATOM'},
        {'symbol': 'LTCUSDT', 'name': 'Litecoin', 'shortName': 'LTC'},
        {'symbol': 'ETCUSDT', 'name': 'Ethereum Classic', 'shortName': 'ETC'},
        {'symbol': 'XLMUSDT', 'name': 'Stellar', 'shortName': 'XLM'},
        {'symbol': 'VETUSDT', 'name': 'VeChain', 'shortName': 'VET'},
        {'symbol': 'AAVEUSDT', 'name': 'Aave', 'shortName': 'AAVE'},
        {'symbol': 'SANDUSDT', 'name': 'The Sandbox', 'shortName': 'SAND'},
        {'symbol': 'MANAUSDT', 'name': 'Decentraland', 'shortName': 'MANA'},
        {'symbol': 'APEUSDT', 'name': 'ApeCoin', 'shortName': 'APE'},
        {'symbol': 'APTUSDT', 'name': 'Aptos', 'shortName': 'APT'},
        {'symbol': 'GALAUSDT', 'name': 'Gala', 'shortName': 'GALA'},
        {'symbol': 'LDOUSDT', 'name': 'Lido DAO', 'shortName': 'LDO'},
        {'symbol': 'CRVUSDT', 'name': 'Curve DAO', 'shortName': 'CRV'},
        {'symbol': 'GMXUSDT', 'name': 'GMX', 'shortName': 'GMX'},
        {'symbol': 'IMXUSDT', 'name': 'Immutable X', 'shortName': 'IMX'},
        {'symbol': 'COMPUSDT', 'name': 'Compound', 'shortName': 'COMP'},
        {'symbol': 'OPUSDT', 'name': 'Optimism', 'shortName': 'OP'},
        {'symbol': 'ARBUSDT', 'name': 'Arbitrum', 'shortName': 'ARB'},
        {'symbol': 'SUIUSDT', 'name': 'Sui', 'shortName': 'SUI'},
        {'symbol': 'NEARUSDT', 'name': 'NEAR Protocol', 'shortName': 'NEAR'},
        {'symbol': 'FTMUSDT', 'name': 'Fantom', 'shortName': 'FTM'}
    ]
    
    try:
        # Verify trading pairs are active
        exchange_info = client.get_exchange_info()
        active_symbols = {symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING'}
        
        # Filter coins to only include active trading pairs
        valid_coins = [coin for coin in verified_coins if coin['symbol'] in active_symbols]
        
        return jsonify(valid_coins)
    except Exception as e:
        print(f"Error getting coins: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '1h')
        
        print(f"Processing request for symbol: {symbol}, interval: {interval}")  # Debug log

        # Validate symbol format
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"
        
        try:
            # Test if the symbol exists
            test_data = client.get_symbol_info(symbol)
            if not test_data:
                raise Exception(f"Invalid symbol: {symbol}")
                
            print(f"Symbol validated: {symbol}")  # Debug log
            
            # Get current time in UTC
            current_time = datetime.utcnow()
            print(f"Current UTC time: {current_time}")

            # Initialize variables
            prediction_length = 0
            description_length = 0
            days_back = 0
            weeks_back = 0
            kline_interval = ''

            # Adjust prediction length and intervals
            if interval == '1h':
                prediction_length = 24
                description_length = 24
                days_back = 2
                kline_interval = '1h'
            elif interval == '1d':
                prediction_length = 5
                description_length = 5
                days_back = 30
                kline_interval = '1d'
            elif interval == '1w':
                prediction_length = 5
                description_length = 35
                weeks_back = 52
                kline_interval = '1w'
            elif interval == '1M':
                prediction_length = 5
                description_length = 150
                days_back = 180
                kline_interval = '1M'
            elif interval == '1y':
                prediction_length = 5
                description_length = 1825
                days_back = 365 * 2
                kline_interval = '1M'

            print(f"Using kline interval: {kline_interval}")

            # Get historical data
            try:
                end_time = int(current_time.timestamp() * 1000)
                if interval == '1w':
                    start_time = end_time - (weeks_back * 7 * 24 * 60 * 60 * 1000)
                else:
                    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
                
                print(f"Fetching historical data from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
                                
                historical_data = client.get_klines(
                                    symbol=symbol,  # Make sure symbol is being used here
                                    interval=kline_interval,
                                    limit=1000,
                                    startTime=start_time,
                                    endTime=end_time
                                )
                
                if not historical_data:
                    raise Exception(f"No historical data found for {symbol}")

                historical_data = [
                    {
                        'timestamp': entry[0],
                        'open': float(entry[1]),
                        'high': float(entry[2]),
                        'low': float(entry[3]),
                        'close': float(entry[4]),
                        'volume': float(entry[5])
                    }
                    for entry in historical_data
                ]

                print(f"Retrieved {len(historical_data)} historical data points")

                # Calculate trends
                price_changes = []
                monthly_patterns = []
                
                for i in range(1, len(historical_data)):
                    prev_close = historical_data[i-1]['close']
                    curr_close = historical_data[i]['close']
                    price_change = (curr_close - prev_close) / prev_close
                    price_changes.append(price_change)
                    
                    if interval == '1w' and len(price_changes) >= 4:
                        monthly_pattern = sum(price_changes[-4:]) / 4
                        monthly_patterns.append(monthly_pattern)

                avg_change = np.mean(price_changes) if price_changes else 0
                volatility = np.std(price_changes) if price_changes else 0.01
                monthly_trend = np.mean(monthly_patterns) if monthly_patterns else avg_change

                # Initialize predictions
                mock_predictions = {
                    'laplace': [],
                    'ml': [],
                    'combined': [],
                    'timestamps': []
                }

                last_price = float(historical_data[-1]['close'])
                base_volatility = max(0.005, min(volatility, 0.03))  # Increased max volatility
                trend = avg_change

                # Generate predictions
                for i in range(prediction_length):
                    # Get current time for each prediction to ensure we're using the latest time
                    current_prediction_time = datetime.utcnow()
                    
                    if interval == '1h':
                        next_date = current_prediction_time + timedelta(hours=i+1)
                    elif interval == '1d':
                        next_date = current_prediction_time + timedelta(days=i+1)
                    elif interval == '1w':
                        next_date = current_prediction_time + timedelta(weeks=i+1)
                    elif interval == '1M':
                        next_date = current_prediction_time + relativedelta(months=i+1)
                    else:  # 1y
                        next_date = current_prediction_time + relativedelta(years=i+1)

                    next_date = next_date.replace(minute=0, second=0, microsecond=0)
                    if interval in ['1d', '1w', '1M', '1y']:
                        next_date = next_date.replace(hour=0)

                    timestamp = int(next_date.timestamp() * 1000)
                    mock_predictions['timestamps'].append(timestamp)
                    
                    print(f"Prediction {i+1} timestamp: {next_date}")  # Debug print
                    
                    # Calculate different predictions for each model
                    period_volatility = base_volatility * (1 + i * 0.15)  # Increased volatility growth
                    
                    # Laplace prediction - more volatile
                    laplace_movement = np.random.normal(trend * 1.2, period_volatility * 1.3)
                    laplace_price = last_price * (1 + laplace_movement)
                    mock_predictions['laplace'].append(laplace_price)
                    
                    # ML prediction - more conservative
                    ml_movement = np.random.normal(trend * 0.8, period_volatility * 0.7)
                    ml_price = last_price * (1 + ml_movement)
                    mock_predictions['ml'].append(ml_price)
                    
                    # Combined prediction - weighted average with some variation
                    combined_price = (laplace_price * 0.4 + ml_price * 0.6) * (1 + np.random.normal(0, 0.002))
                    mock_predictions['combined'].append(combined_price)
                    
                    # Update last price using the combined prediction
                    last_price = combined_price

                    # Add weekly pattern if applicable
                    if interval == '1w':
                        week_in_month = (i % 4) + 1
                        if week_in_month == 1:  # First week typically up
                            mock_predictions['laplace'][-1] *= 1.02
                            mock_predictions['ml'][-1] *= 1.01
                            mock_predictions['combined'][-1] *= 1.015
                        elif week_in_month == 2:  # Second week consolidation
                            mock_predictions['laplace'][-1] *= 0.99
                            mock_predictions['ml'][-1] *= 0.995
                            mock_predictions['combined'][-1] *= 0.992
                        elif week_in_month == 3:  # Third week typically down
                            mock_predictions['laplace'][-1] *= 0.98
                            mock_predictions['ml'][-1] *= 0.99
                            mock_predictions['combined'][-1] *= 0.985
                        else:  # Fourth week recovery
                            mock_predictions['laplace'][-1] *= 1.015
                            mock_predictions['ml'][-1] *= 1.005
                            mock_predictions['combined'][-1] *= 1.01

                # Verify predictions
                if not all(len(mock_predictions[key]) == prediction_length for key in ['laplace', 'ml', 'combined', 'timestamps']):
                    raise Exception("Prediction arrays have inconsistent lengths")

                print(f"Generated predictions: {len(mock_predictions['timestamps'])} timestamps")

                description_prediction = {
                    'next_value': mock_predictions['combined'][-1],
                    'timeframe': description_length,
                    'graph_timeframe': prediction_length,
                    'end_timestamp': mock_predictions['timestamps'][-1]
                }
                    
                response_data = {
                    'historical_data': historical_data,
                    'predictions': mock_predictions,
                    'description_prediction': description_prediction
                }
                
                return jsonify(response_data)

            except Exception as e:
                print(f"Error in prediction generation: {str(e)}")
                print("Traceback:", traceback.format_exc())
                raise Exception(f"Failed to generate predictions: {str(e)}")
            
        except Exception as e:
            print(f"Error validating symbol {symbol}: {str(e)}")  # Debug log
            raise Exception(f"Invalid symbol: {symbol}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/visualization')
def visualization():
    return render_template('data_visualization.html')

# Also add the historical prices endpoint
@app.route('/api/historical_prices', methods=['POST'])
def get_historical_prices():
    try:
        # Validate input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symbol = data.get('symbol')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        if not all([symbol, start_date, end_date]):
            return jsonify({'error': 'Missing required parameters'}), 400

        print(f"Fetching historical prices for {symbol} from {start_date} to {end_date}")

        # Ensure symbol ends with USDT
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        try:
            # Convert dates to millisecond timestamps for Binance API
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

            # Get historical klines from Binance with retry
            attempts = 3
            klines = None
            
            # Try different time windows if the initial request fails
            time_windows = [
                (start_ts, end_ts),  # Original time window
                ("6 months ago UTC", None),  # Last 6 months
                ("3 months ago UTC", None),  # Last 3 months
                ("1 month ago UTC", None)    # Last month
            ]

            for time_window in time_windows:
                for attempt in range(attempts):
                    try:
                        if isinstance(time_window[0], str):
                            # Using string-based time window
                            klines = client.get_historical_klines(
                                symbol,
                                Client.KLINE_INTERVAL_1DAY,
                                time_window[0],
                                time_window[1]
                            )
                        else:
                            # Using timestamp-based time window
                            klines = client.get_historical_klines(
                                symbol,
                                Client.KLINE_INTERVAL_1DAY,
                                start_str=time_window[0],
                                end_str=time_window[1]
                            )
                        
                        if klines:
                            print(f"Successfully fetched data using time window: {time_window}")
                            break
                        time.sleep(1)
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed for time window {time_window}: {str(e)}")
                        if attempt == attempts - 1:
                            continue
                        time.sleep(1)
                
                if klines:
                    break

            if not klines:
                raise Exception(f"No data available for {symbol} in any time window")

            # Format the data and filter for 10-day intervals
            historical_data = []
            for i, kline in enumerate(klines):
                if i % 10 == 0:  # Only keep every 10th day
                    try:
                        timestamp = datetime.fromtimestamp(kline[0] / 1000)
                        close_price = float(kline[4])
                        
                        if close_price > 0:  # Validate price
                            historical_data.append({
                                'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                                'close': close_price
                            })
                    except (ValueError, IndexError) as e:
                        print(f"Error processing kline: {str(e)}")
                        continue

            if not historical_data:
                return jsonify({'error': f'No data available for {symbol}'}), 404

            # Sort data by timestamp
            historical_data.sort(key=lambda x: x['timestamp'])
            
            # Filter to requested date range if using fallback data
            if isinstance(time_windows[0], str):
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                historical_data = [
                    data for data in historical_data 
                    if start_date_obj <= datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%S') <= end_date_obj
                ]

            print(f"Retrieved {len(historical_data)} data points for {symbol}")
            return jsonify(historical_data)

        except Exception as e:
            print(f"Error fetching from Binance: {str(e)}")
            return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(error_msg)
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/actual_prices', methods=['POST'])
def get_actual_prices():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        print(f"Fetching actual prices for {symbol} from {start_date} to {end_date}")

        # Ensure symbol ends with USDT
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        try:
            # Convert dates to millisecond timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

            # Get actual prices from Binance
            klines = client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1DAY,
                start_str=start_ts,
                end_str=end_ts
            )

            if not klines:
                return jsonify([])

            # Format the data
            actual_prices = []
            for kline in klines:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                close_price = float(kline[4])
                
                if close_price > 0:
                    actual_prices.append({
                        'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                        'close': close_price
                    })

            return jsonify(actual_prices)

        except Exception as e:
            print(f"Error fetching actual prices: {str(e)}")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        print(f"Error in get_actual_prices: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/coin_info/<symbol>')
def get_coin_info(symbol):
    try:
        # Remove USDT suffix if present
        base_symbol = symbol.replace('USDT', '')
        
        # Get CoinGecko ID from the mapping
        coingecko_id = COINGECKO_ID_MAP.get(base_symbol)
        if not coingecko_id:
            return jsonify({'error': 'Unsupported symbol'}), 400

        # Add retry mechanism for CoinGecko API
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch coin data from CoinGecko
                coin_data = cg.get_coin_by_id(
                    coingecko_id,
                    localization=False,
                    tickers=False,
                    market_data=True,
                    community_data=False,
                    developer_data=False,
                    sparkline=False
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Wait before retry

        # Extract relevant data with default values
        market_data = coin_data.get('market_data', {})
        
        response_data = {
            'market_data': {
                'market_cap_rank': coin_data.get('market_cap_rank', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'total_supply': market_data.get('total_supply', 0),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'atl': market_data.get('atl', {}).get('usd', 0),
                'ath_date': market_data.get('ath_date', {}).get('usd', ''),
                'atl_date': market_data.get('atl_date', {}).get('usd', '')
            },
            'current_data': {
                'price': market_data.get('current_price', {}).get('usd', 0),
                'price_change_percentage_24h': market_data.get('price_change_percentage_24h', 0)
            }
        }

        # Validate response data
        if not response_data['current_data']['price']:
            raise Exception('Invalid price data received')
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error fetching coin info: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_prediction_summary(historical_data, predictions, interval):
    """Generate a comprehensive prediction summary"""
    
    # Calculate price movement percentage
    last_price = historical_data[-1]['close']
    predicted_price = predictions['combined'][-1]
    price_movement = ((predicted_price - last_price) / last_price) * 100

    # Determine trend
    trend = 'BULLISH' if price_movement > 0 else 'BEARISH'

    # Calculate confidence based on multiple factors
    confidence = calculate_prediction_confidence(
        historical_data=historical_data,
        predictions=predictions,
        price_movement=price_movement,
        interval=interval
    )

    # Generate time frame description
    timeframe_map = {
        '1h': 'hour',
        '1d': 'day',
        '1w': 'week',
        '1M': 'month',
        '1y': 'year'
    }
    
    # Generate recommendation
    recommendation = generate_recommendation(confidence, price_movement)

    return {
        'trend': trend,
        'price_movement': abs(price_movement),
        'confidence': confidence,
        'timeframe': timeframe_map.get(interval, 'period'),
        'recommendation': recommendation
    }

def calculate_prediction_confidence(historical_data, predictions, price_movement, interval):
    """Calculate the confidence score (0-100)"""
    
    # 1. Historical Volatility Impact (25%)
    returns = np.diff([d['close'] for d in historical_data]) / [d['close'] for d in historical_data[:-1]]
    volatility = np.std(returns)
    volatility_score = min(25, (1 - volatility) * 25)  # Lower volatility = higher confidence

    # 2. Prediction Consistency (25%)
    prediction_changes = np.diff(predictions['combined']) / predictions['combined'][:-1]
    consistency_score = min(25, (1 - np.std(prediction_changes)) * 25)

    # 3. Market Trend Strength (25%)
    prices = [d['close'] for d in historical_data]
    trend_strength = calculate_trend_strength(prices)
    trend_score = trend_strength * 25

    # 4. Price Movement Reasonability (25%)
    movement_score = calculate_movement_reasonability(price_movement, interval)

    # Calculate total confidence
    total_confidence = int(volatility_score + consistency_score + trend_score + movement_score)
    
    # Ensure confidence is between 0 and 100
    return max(0, min(100, total_confidence))

def calculate_trend_strength(prices):
    """Calculate the strength of the market trend"""
    x = np.arange(len(prices))
    y = np.array(prices)
    
    # Calculate R² value of linear regression
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    return abs(r_squared)

def calculate_movement_reasonability(movement, interval):
    """Calculate how reasonable the predicted price movement is"""
    # Define maximum reasonable movements for each interval
    max_movements = {
        '1h': 5,    # 5% per hour
        '1d': 15,   # 15% per day
        '1w': 40,   # 40% per week
        '1M': 100,  # 100% per month
        '1y': 300   # 300% per year
    }
    
    max_movement = max_movements.get(interval, 50)
    movement_abs = abs(movement)
    
    if movement_abs > max_movement:
        return 25 * (max_movement / movement_abs)
    return 25

def generate_recommendation(confidence, price_movement):
    """Generate investment recommendation based on confidence and movement"""
    if confidence >= 75:
        return "Strong signal for investment decision. Consider acting based on your strategy."
    elif confidence >= 60:
        return "Consider accumulating based on your risk tolerance and investment strategy."
    elif confidence >= 40:
        return "Monitor the situation closely before making any decisions."
    else:
        return "High uncertainty. Consider waiting for clearer signals."

if __name__ == '__main__':
    app.run(debug=True, port=8000)