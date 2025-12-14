import yfinance as yf

df = yf.download("GC=F", period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
print(df.shape, df.head(), df.tail())