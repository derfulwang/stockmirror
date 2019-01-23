import tushare as ts
import statsmodels.api as sm
df = ts.get_k_data("512880")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df = df.set_index("date")
industry = ts.get_industry_classified()
finance = industry[industry.c_name=='金融行业']
pfyh = ts.get_k_data("600000")

pfyh["date"] = pd.to_datetime(pfyh["date"], format="%Y-%m-%d")
pfyh = pfyh.set_index("date")
res = sm.tsa.coint(df.close, pfyh['2016-08-08':].close)

df.close.plot(secondary_y=True)
pfyh['2016-08-08':].close.plot()


finan_df = []
for code in finance.code.values:
    tmp = ts.get_k_data(code)
    tmp["date"] = pd.to_datetime(tmp["date"], format="%Y-%m-%d")
    tmp = tmp.set_index("date")
    tmp = tmp['2016-08-08':]
    finan_df.append(tmp)
finan_df = pd.concat(finan_df)

res = {}
for code in finance.code.values:
    tmp = finan_df[finan_df.code==code]
    if tmp.shape[0] != df.shape[0]:
        print(code)
        continue
    res[code] = sm.tsa.coint(df.close, tmp.close)

r = sorted(res.items(), key=lambda x:x[1][1])
df.close.plot(secondary_y=True,label='512880')
pfyh['2016-08-08':].close.plot(label="600000")
plt.legend()


x = df.close
X = sm.add_constant(x)
y = finan_df[finan_df.code=='601688'].close
result = (sm.OLS(y,X)).fit()
print(result.summary())

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, result.fittedvalues, 'r', label="OLS")
ax.legend(loc='best')

plt.plot(10.6284*x-y)
plt.axhline((10.6284*x-y).mean(), color="red", linestyle="--")
plt.xlabel("Time"); plt.ylabel("Stationary Series")
plt.legend(["Stationary Series", "Mean"])

def zscore(series):
    return (series - series.mean()) / np.std(series)

plt.plot(zscore(7.2836*x-y))
plt.axhline(zscore(7.2836*x-y).mean(), color="black")
plt.axhline(1.0, color="red", linestyle="--")
plt.axhline(-1.0, color="green", linestyle="--")
plt.legend(["z-score", "mean", "+1", "-1"])