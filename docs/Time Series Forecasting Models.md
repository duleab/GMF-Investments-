#### GMF Investments - Task 2: Time Series Forecasting Models
#### Building Predictive Models for TSLA Price Forecasting
**Models to Implement**:
1. Statistical: ARIMA/SARIMA
2. Deep Learning: LSTM Neural Networks

**Evaluation Period**:
- Training: 2015-2023
- Testing: 2024-2025


### ENVIRONMENT SETUP & ADVANCED LIBRARIES

    Requirement already satisfied: yfinance in /usr/local/lib/python3.11/dist-packages (0.2.65)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (2.0.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (3.10.0)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.11/dist-packages (0.13.2)
    Requirement already satisfied: plotly in /usr/local/lib/python3.11/dist-packages (5.24.1)
    Requirement already satisfied: requests>=2.31 in /usr/local/lib/python3.11/dist-packages (from yfinance) (2.32.3)
    Requirement already satisfied: multitasking>=0.0.7 in /usr/local/lib/python3.11/dist-packages (from yfinance) (0.0.12)
    Requirement already satisfied: platformdirs>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from yfinance) (4.3.8)
    Requirement already satisfied: pytz>=2022.5 in /usr/local/lib/python3.11/dist-packages (from yfinance) (2025.2)
    Requirement already satisfied: frozendict>=2.3.4 in /usr/local/lib/python3.11/dist-packages (from yfinance) (2.4.6)
    Requirement already satisfied: peewee>=3.16.2 in /usr/local/lib/python3.11/dist-packages (from yfinance) (3.18.2)
    Requirement already satisfied: beautifulsoup4>=4.11.1 in /usr/local/lib/python3.11/dist-packages (from yfinance) (4.13.4)
    Requirement already satisfied: curl_cffi>=0.7 in /usr/local/lib/python3.11/dist-packages (from yfinance) (0.13.0)
    Requirement already satisfied: protobuf>=3.19.0 in /usr/local/lib/python3.11/dist-packages (from yfinance) (5.29.5)
    Requirement already satisfied: websockets>=13.0 in /usr/local/lib/python3.11/dist-packages (from yfinance) (15.0.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (4.59.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (25.0)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (3.2.3)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.11/dist-packages (from plotly) (8.5.0)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4>=4.11.1->yfinance) (2.7)
    Requirement already satisfied: typing-extensions>=4.0.0 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4>=4.11.1->yfinance) (4.14.1)
    Requirement already satisfied: cffi>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from curl_cffi>=0.7->yfinance) (1.17.1)
    Requirement already satisfied: certifi>=2024.2.2 in /usr/local/lib/python3.11/dist-packages (from curl_cffi>=0.7->yfinance) (2025.8.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.31->yfinance) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.31->yfinance) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.31->yfinance) (2.5.0)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.11/dist-packages (from cffi>=1.12.0->curl_cffi>=0.7->yfinance) (2.22)
    Requirement already satisfied: statsmodels in /usr/local/lib/python3.11/dist-packages (0.14.5)
    Collecting pmdarima
      Downloading pmdarima-2.0.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (7.8 kB)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (1.6.1)
    Requirement already satisfied: numpy<3,>=1.22.3 in /usr/local/lib/python3.11/dist-packages (from statsmodels) (2.0.2)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in /usr/local/lib/python3.11/dist-packages (from statsmodels) (1.16.1)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in /usr/local/lib/python3.11/dist-packages (from statsmodels) (2.2.2)
    Requirement already satisfied: patsy>=0.5.6 in /usr/local/lib/python3.11/dist-packages (from statsmodels) (1.0.1)
    Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.11/dist-packages (from statsmodels) (25.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.11/dist-packages (from pmdarima) (1.5.1)
    Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in /usr/local/lib/python3.11/dist-packages (from pmdarima) (3.0.12)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.11/dist-packages (from pmdarima) (2.5.0)
    Requirement already satisfied: setuptools!=50.0.0,>=38.6.0 in /usr/local/lib/python3.11/dist-packages (from pmdarima) (75.2.0)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas!=2.1.0,>=1.4->statsmodels) (1.17.0)
    Downloading pmdarima-2.0.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m37.0 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pmdarima
    Successfully installed pmdarima-2.0.4
    Requirement already satisfied: tensorflow in /usr/local/lib/python3.11/dist-packages (2.19.0)
    Requirement already satisfied: keras in /usr/local/lib/python3.11/dist-packages (3.10.0)
    Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.4.0)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.2.10)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.6.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (18.1.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.4.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (5.29.5)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.32.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from tensorflow) (75.2.0)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.1.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.14.1)
    Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.2)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.74.0)
    Requirement already satisfied: tensorboard~=2.19.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.19.0)
    Requirement already satisfied: numpy<2.2.0,>=1.26.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.0.2)
    Requirement already satisfied: h5py>=3.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.14.0)
    Requirement already satisfied: ml-dtypes<1.0.0,>=0.5.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.5.3)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.37.1)
    Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras) (13.9.4)
    Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras) (0.1.0)
    Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras) (0.17.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2025.8.3)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard~=2.19.0->tensorflow) (3.8.2)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard~=2.19.0->tensorflow) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard~=2.19.0->tensorflow) (3.1.3)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras) (0.1.2)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from werkzeug>=1.0.1->tensorboard~=2.19.0->tensorflow) (3.0.2)
    [31mERROR: Could not find a version that satisfies the requirement warnings-ignore (from versions: none)[0m[31m
    [0m[31mERROR: No matching distribution found for warnings-ignore[0m[31m
    [0m

     Advanced forecasting environment ready!
     TensorFlow version: 2.19.0
    

#### DATA PREPARATION & FEATURE ENGINEERING

     Loading TSLA data...
    Downloaded data columns: [('Close', 'TSLA'), ('High', 'TSLA'), ('Low', 'TSLA'), ('Open', 'TSLA'), ('Volume', 'TSLA')]
     'Adj Close' not found, using 'Close' instead.
     Data has multi-level columns. Flattening...
    Flattened columns: ['Close', 'High', 'Low', 'Open', 'Volume']
    Data prepared: 2486 records with 15 features
     Date range: 2015-09-10 to 2025-07-30
    
     Dataset Overview:
    --------------------------------------------------
    Shape: (2486, 15)
    
    Features: ['Close', 'High', 'Low', 'Open', 'Volume', 'Returns', 'Price_MA_5', 'Price_MA_20', 'Price_MA_50', 'Volatility', 'High_Low_Pct', 'Volume_MA', 'Volume_Ratio', 'Price_Change_5d', 'Price_Change_20d']
    
    Sample data:
    Price           Close       High        Low       Open    Volume   Returns  \
    Date                                                                         
    2015-09-10  16.565332  16.714666  16.355333  16.482000  40635000 -0.001729   
    2015-09-11  16.682667  16.682667  16.315332  16.509333  35262000  0.007058   
    2015-09-14  16.879333  16.950001  16.644667  16.740000  43363500  0.011720   
    2015-09-15  16.904667  16.973333  16.633333  16.850000  44002500  0.001500   
    2015-09-16  17.483334  17.525333  16.858667  16.869333  66256500  0.033658   
    
    Price       Price_MA_5  Price_MA_20  Price_MA_50  Volatility  High_Low_Pct  \
    Date                                                                         
    2015-09-10   16.440800    16.177400    16.960173    0.546823      0.021970   
    2015-09-11   16.503067    16.203166    16.934960    0.543957      0.022515   
    2015-09-14   16.653200    16.236633    16.899187    0.545148      0.018344   
    2015-09-15   16.725200    16.231900    16.864320    0.517975      0.020441   
    2015-09-16   16.903067    16.237000    16.856813    0.526091      0.039544   
    
    Price        Volume_MA  Volume_Ratio  Price_Change_5d  Price_Change_20d  
    Date                                                                     
    2015-09-10  74605350.0      0.544666         0.003189          0.043288  
    2015-09-11  72851550.0      0.484025         0.019017          0.031875  
    2015-09-14  71746125.0      0.604402         0.046542          0.041291  
    2015-09-15  68563725.0      0.641775         0.021759         -0.005569  
    2015-09-16  68730300.0      0.964007         0.053594          0.005868  
    

#### DATA SPLITTING & VISUALIZATION

     Data Split Summary:
    Training Period: 2015-09-10 to 2023-12-29
    Testing Period: 2024-01-02 to 2025-07-30
    Training samples: 2091
    Testing samples: 395
    




#### ARIMA MODEL DEVELOPMENT

     Fitting ARIMA model...
     Searching for optimal ARIMA parameters...
    Performing stepwise search to minimize aic
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=13383.216, Time=0.08 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=13383.275, Time=0.10 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=13383.328, Time=0.23 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=13381.945, Time=0.05 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=13383.758, Time=0.82 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0]          
    Total fit time: 1.309 seconds
     Optimal ARIMA order: (0, 1, 0)
     ARIMA model fitted successfully!
     Model Summary:
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  Close   No. Observations:                 2091
    Model:                 ARIMA(0, 1, 0)   Log Likelihood               -6689.972
    Date:                Sun, 10 Aug 2025   AIC                          13381.945
    Time:                        17:05:17   BIC                          13387.590
    Sample:                             0   HQIC                         13384.013
                                   - 2091                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    sigma2        35.3052      0.433     81.577      0.000      34.457      36.153
    ===================================================================================
    Ljung-Box (L1) (Q):                   1.94   Jarque-Bera (JB):             10084.20
    Prob(Q):                              0.16   Prob(JB):                         0.00
    Heteroskedasticity (H):             432.05   Skew:                            -0.16
    Prob(H) (two-sided):                  0.00   Kurtosis:                        13.76
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


    
![png](Time%20Series%20Forecasting%20Models_files/Time%20Series%20Forecasting%20Models_9_1.png)
    


     Generated 395 ARIMA forecasts
    

#### LSTM MODEL DEVELOPMENT

     Preparing data for LSTM...
     LSTM data prepared:
    Training sequences: (2031, 60)
    Training targets: (2031,)
    Training LSTM model...
     Building LSTM model...
     LSTM model built successfully!
     Model Summary:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">10,400</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">20,200</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">20,200</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">25</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,275</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">26</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">52,101</span> (203.52 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">52,101</span> (203.52 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    Epoch 1/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 21ms/step - loss: 0.0255 - mae: 0.0894 - val_loss: 0.0034 - val_mae: 0.0466
    Epoch 2/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 14ms/step - loss: 0.0024 - mae: 0.0269 - val_loss: 0.0045 - val_mae: 0.0567
    Epoch 3/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0020 - mae: 0.0252 - val_loss: 0.0025 - val_mae: 0.0409
    Epoch 4/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0020 - mae: 0.0248 - val_loss: 0.0015 - val_mae: 0.0311
    Epoch 5/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 0.0017 - mae: 0.0228 - val_loss: 0.0013 - val_mae: 0.0295
    Epoch 6/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 0.0017 - mae: 0.0222 - val_loss: 0.0014 - val_mae: 0.0297
    Epoch 7/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 0.0014 - mae: 0.0216 - val_loss: 0.0013 - val_mae: 0.0299
    Epoch 8/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 0.0016 - mae: 0.0209 - val_loss: 0.0011 - val_mae: 0.0269
    Epoch 9/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0013 - mae: 0.0198 - val_loss: 0.0017 - val_mae: 0.0347
    Epoch 10/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0012 - mae: 0.0197 - val_loss: 0.0014 - val_mae: 0.0314
    Epoch 11/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0013 - mae: 0.0202 - val_loss: 0.0011 - val_mae: 0.0265
    Epoch 12/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 9.6070e-04 - mae: 0.0185 - val_loss: 9.1469e-04 - val_mae: 0.0241
    Epoch 13/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 0.0010 - mae: 0.0189 - val_loss: 0.0011 - val_mae: 0.0257
    Epoch 14/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0011 - mae: 0.0198 - val_loss: 8.1022e-04 - val_mae: 0.0223
    Epoch 15/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 0.0010 - mae: 0.0192 - val_loss: 0.0016 - val_mae: 0.0319
    Epoch 16/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - loss: 0.0012 - mae: 0.0207 - val_loss: 0.0018 - val_mae: 0.0349
    Epoch 17/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 20ms/step - loss: 0.0011 - mae: 0.0199 - val_loss: 7.4883e-04 - val_mae: 0.0215
    Epoch 18/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 8.9321e-04 - mae: 0.0184 - val_loss: 8.7350e-04 - val_mae: 0.0237
    Epoch 19/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 8.9863e-04 - mae: 0.0180 - val_loss: 8.6392e-04 - val_mae: 0.0233
    Epoch 20/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 8.9173e-04 - mae: 0.0181 - val_loss: 0.0017 - val_mae: 0.0345
    Epoch 21/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 9.6148e-04 - mae: 0.0197 - val_loss: 7.6641e-04 - val_mae: 0.0220
    Epoch 22/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 8.5080e-04 - mae: 0.0181 - val_loss: 0.0010 - val_mae: 0.0255
    Epoch 23/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.8709e-04 - mae: 0.0176 - val_loss: 6.5010e-04 - val_mae: 0.0202
    Epoch 24/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.4208e-04 - mae: 0.0173 - val_loss: 6.0272e-04 - val_mae: 0.0193
    Epoch 25/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 8.4597e-04 - mae: 0.0185 - val_loss: 6.7184e-04 - val_mae: 0.0205
    Epoch 26/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 9.0042e-04 - mae: 0.0188 - val_loss: 6.1305e-04 - val_mae: 0.0195
    Epoch 27/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.9698e-04 - mae: 0.0171 - val_loss: 5.5479e-04 - val_mae: 0.0187
    Epoch 28/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.0009e-04 - mae: 0.0171 - val_loss: 7.7793e-04 - val_mae: 0.0223
    Epoch 29/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - loss: 6.5599e-04 - mae: 0.0166 - val_loss: 8.7845e-04 - val_mae: 0.0237
    Epoch 30/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - loss: 8.0035e-04 - mae: 0.0187 - val_loss: 8.1723e-04 - val_mae: 0.0230
    Epoch 31/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.9562e-04 - mae: 0.0182 - val_loss: 7.2495e-04 - val_mae: 0.0216
    Epoch 32/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 7.2940e-04 - mae: 0.0177 - val_loss: 6.6229e-04 - val_mae: 0.0203
    Epoch 33/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.6660e-04 - mae: 0.0174 - val_loss: 5.0811e-04 - val_mae: 0.0179
    Epoch 34/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 16ms/step - loss: 5.7391e-04 - mae: 0.0163 - val_loss: 5.4647e-04 - val_mae: 0.0187
    Epoch 35/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - loss: 6.2712e-04 - mae: 0.0167 - val_loss: 5.5517e-04 - val_mae: 0.0188
    Epoch 36/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.2316e-04 - mae: 0.0175 - val_loss: 4.9666e-04 - val_mae: 0.0177
    Epoch 37/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.5890e-04 - mae: 0.0170 - val_loss: 8.8551e-04 - val_mae: 0.0241
    Epoch 38/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 7.7502e-04 - mae: 0.0186 - val_loss: 5.1796e-04 - val_mae: 0.0176
    Epoch 39/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.1679e-04 - mae: 0.0165 - val_loss: 6.0006e-04 - val_mae: 0.0191
    Epoch 40/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 5.3819e-04 - mae: 0.0160 - val_loss: 4.5328e-04 - val_mae: 0.0166
    Epoch 41/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 20ms/step - loss: 5.4887e-04 - mae: 0.0158 - val_loss: 4.5144e-04 - val_mae: 0.0168
    Epoch 42/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 5.7107e-04 - mae: 0.0162 - val_loss: 4.7220e-04 - val_mae: 0.0173
    Epoch 43/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 6.4563e-04 - mae: 0.0169 - val_loss: 4.3780e-04 - val_mae: 0.0162
    Epoch 44/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 6.2988e-04 - mae: 0.0172 - val_loss: 4.5231e-04 - val_mae: 0.0168
    Epoch 45/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 6.2005e-04 - mae: 0.0170 - val_loss: 4.5905e-04 - val_mae: 0.0165
    Epoch 46/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 6.8017e-04 - mae: 0.0187 - val_loss: 5.4230e-04 - val_mae: 0.0186
    Epoch 47/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.1258e-04 - mae: 0.0166 - val_loss: 4.1631e-04 - val_mae: 0.0159
    Epoch 48/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 5.8858e-04 - mae: 0.0164 - val_loss: 4.1704e-04 - val_mae: 0.0159
    Epoch 49/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step - loss: 6.8116e-04 - mae: 0.0175 - val_loss: 5.4085e-04 - val_mae: 0.0186
    Epoch 50/50
    [1m58/58[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step - loss: 6.3865e-04 - mae: 0.0176 - val_loss: 4.0389e-04 - val_mae: 0.0154
     LSTM training completed!
    


    
![png](Time%20Series%20Forecasting%20Models_files/Time%20Series%20Forecasting%20Models_11_7.png)
    


     Generating 395 LSTM predictions...
     Generated 395 LSTM predictions
    

#### MODEL EVALUATION & COMPARISON

     MODEL PERFORMANCE EVALUATION
    ================================================================================
    
     Performance Metrics Comparison:
    ------------------------------------------------------------
    Metric               ARIMA           LSTM            Winner    
    ------------------------------------------------------------
    MAE                  62.970          64.309          ARIMA     
    RMSE                 77.963          81.885          ARIMA     
    MAPE                 24.089          23.362          LSTM      
    Directional_Accuracy 50.508          50.508          LSTM      
    

#### FORECAST VISUALIZATION




#### RESIDUAL ANALYSIS & MODEL INSIGHTS


    
![png](Time%20Series%20Forecasting%20Models_files/Time%20Series%20Forecasting%20Models_17_0.png)
    


    
     RESIDUAL STATISTICS:
    ==================================================
    ARIMA Residuals:
      Mean: 13.3209
      Std:  76.8166
      Skew: 0.6448
      Kurt: -0.5368
    
    LSTM Residuals:
      Mean: 26.3307
      Std:  77.5359
      Skew: 0.6402
      Kurt: -0.5659
    

#### MODEL INTERPRETATION & INSIGHTS

     GENERATING TRADING SIGNALS
    ============================================================
     Signal Summary:
    Total predictions: 395
    Models agree: 325 times (82.3%)
    
     ARIMA Signal Distribution:
    ARIMA_Action
    Strong Buy     178
    Strong Sell    162
    Hold            25
    Buy             18
    Sell            12
    Name: count, dtype: int64
    
     LSTM Signal Distribution:
    LSTM_Action
    Strong Sell    194
    Strong Buy     164
    Hold            15
    Sell            13
    Buy              9
    Name: count, dtype: int64
    

#### FORECAST UNCERTAINTY ANALYSIS

     FORECAST UNCERTAINTY ANALYSIS
    ============================================================
     ARIMA Confidence Intervals:
    Average CI width: $309.18
    CI width std: $108.86
    Max CI width: $462.91
    Min CI width: $23.29
    Coverage probability: 94.4%
    
     LSTM Uncertainty Estimation:
    Prediction error std: $50.69
    Mean absolute error: $64.31
    


    
![png](Time%20Series%20Forecasting%20Models_files/Time%20Series%20Forecasting%20Models_21_1.png)
    


#### MODEL SELECTION & RECOMMENDATIONS

     MODEL SELECTION & RECOMMENDATIONS
    ================================================================================
     WINNING MODEL: TIE
       ARIMA Score: 2/4
       LSTM Score: 2/4
    
     INVESTMENT RECOMMENDATIONS:
    --------------------------------------------------
    Current TSLA Price: $319.04
    12-Month Forecast: $241.49
    Expected Return: -24.3%
    
     PORTFOLIO ALLOCATION GUIDANCE:
    Recommendation: REDUCE
    Suggested Allocation: 0-3%
    Risk Level: HIGH
    Confidence: Medium
    
     MODEL-SPECIFIC INSIGHTS:
    --------------------------------------------------
    • Models show comparable performance
    • Ensemble approach recommended
    • Combination reduces individual model risk
    • Consider balanced technical/fundamental analysis
    

#### FORECAST EXPORT & PREPARATION FOR TASK 3

     PREPARING DATA FOR TASK 3
    ==================================================
     TSLA Forecast Summary for Portfolio Optimization:
       Winning Model: ENSEMBLE
       Current Price: $248.42
       Forecast Price: $241.49
       Expected Annual Return: -1.79%
       Forecast Horizon: 395 days
    
     Forecast dataset prepared: 395 predictions
    

#### COMPREHENSIVE TASK 2 SUMMARY

     TASK 2 COMPLETION SUMMARY
    ================================================================================
     COMPLETED DELIVERABLES:
    --------------------------------------------------
    1.  ARIMA Model Development
       • Optimal parameter identification using auto_arima
       • Model fitting and validation
       • Confidence interval generation
    
    2.  LSTM Neural Network Implementation
       • Data preprocessing and sequence creation
       • Multi-layer LSTM architecture
       • Training with early stopping
       • Prediction generation
    
    3.  Model Evaluation & Comparison
       • MAE, RMSE, MAPE calculations
       • Directional accuracy assessment
       • Residual analysis
       • Statistical significance testing
    
    4.  Forecast Visualization
       • Interactive plotly charts
       • Confidence interval visualization
       • Prediction vs actual comparison
       • Uncertainty band analysis
    
    5.  Trading Signal Generation
       • Buy/sell/hold signal categorization
       • Model agreement analysis
       • Risk-adjusted recommendations
    
     KEY FINDINGS:
    ------------------------------
    • Winning Model: ENSEMBLE
    • Expected Return: -24.3%
    • Investment Recommendation: REDUCE
    • Suggested Allocation: 0-3%
    • Model Confidence: Medium
    
     PERFORMANCE METRICS SUMMARY:
    ----------------------------------------
    ARIMA - MAE: 62.97, RMSE: 77.96
    LSTM  - MAE: 64.31, RMSE: 81.88
    
    READY FOR TASK 3:
    ------------------------------
    • TSLA expected return calculated
    • Model confidence assessed
    • Risk metrics available
    • Portfolio optimization inputs prepared
    

#### ADVANCED INSIGHTS & MARKET IMPLICATIONS

     ADVANCED MARKET INSIGHTS
    ============================================================
     TREND ANALYSIS:
    Daily trend coefficient: -0.0065
    Monthly trend: -0.14 $/month
    Annual trend: -1.64 $/year
    
    ⚡ VOLATILITY FORECAST:
    Predicted volatility: 0.6%
    Historical volatility: 30.0%
    
     MARKET REGIME ANALYSIS:
    Bullish days expected: 0.0%
    Market sentiment: Bearish
    
     RISK-ADJUSTED EXPECTATIONS:
    Forecasted Sharpe ratio: -42.475
    Risk category: Low
    
     STRATEGIC RECOMMENDATIONS FOR GMF:
    --------------------------------------------------
    • Reduce TSLA exposure
    • Consider defensive positioning
    • Explore alternative growth assets
    
     NEXT STEPS FOR PORTFOLIO CONSTRUCTION:
    • Integrate TSLA forecast into MPT optimization
    • Calculate correlation with BND and SPY
    • Determine optimal portfolio weights
    • Design rebalancing strategy
    

