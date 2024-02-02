# ポートフォリオ最適化
import sys
import requests
from io import StringIO
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = "Yu Gothic"

from pymoo.core.problem import ElementwiseProblem
import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# Yahoo financeから、URLでjsonデータ取得
def get_json_from_yf(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    str = StringIO(response.text)
    j = json.load(str)
    return j

# ティッカーシンボル指定で株価(1日の終値)と分配金情報を取得
def get_quote_dividend(ticker, period, interval):
    # urlを作成しjsonを取得
    prefix = "https://query1.finance.yahoo.com/v8/finance/chart/"
    url = prefix + ticker + "?range=" +period + "&interval=" + interval + "&events=div&includeAdjustedClose=true"
    j = get_json_from_yf(url)

    # 株価の取得
    quote = pd.DataFrame()
    timestamp = j['chart']['result'][0]['timestamp']
    quote.index = pd.to_datetime(timestamp, unit="s")
    quote.index.name = "timestamp"
    quote['close'] = j['chart']['result'][0]['indicators']['quote'][0]['close']

    # 分配金情報の取得
    try:
        div_json = j["chart"]["result"][0]["events"]["dividends"]
        # 辞書の値のリストと列名のリストをDataFrameのコンストラクタに渡す
        div = pd.DataFrame(list(div_json.values()), columns=['amount', 'date'])
        div.index = pd.to_datetime(div["date"], unit="s")
        div.index.name = "timestamp"
        div = div.drop("date", axis=1)
        div.columns = ["dividend"]
    except:
        div = None

    return quote, div

# USDの株価をJPYに変換
# 入力の列名は"timestamp","close"
def conv_usd_to_jpy(price, usdjpy):
    # stockのtimestampに最も近いusdjpyのtimestampを結合
    merged = pd.merge_asof(price, usdjpy, on="timestamp", direction="nearest")
    # ドル建ての価格を円に換算
    merged[price.columns[0]] = merged[merged.columns[1]] * merged[merged.columns[2]]
    # timestampをindexにする
    merged = merged.set_index("timestamp")

    return pd.DataFrame(merged[price.columns[0]])

# 株価データを過去10年分データに整形する。休日は翌営業日データで取引されると考え、bfillでデータを埋める
def bfill_10y(df):
    new_index = pd.date_range(start="2013-12-31", end="2024-01-10") # 新しい日付範囲
    new_index.name = df.index.name
    df = df.reindex(new_index) # 新しいインデックスを設定
    df = df.bfill() # 欠損値を後方から埋める
    
    return df

# 株価取得
def get_prices():
    # ティッカーシンボル一覧
    ticker_list = ["CASH", "1306.T", "SPY","VEA","VWO","BND","BWX","1343.T","VNQ","IFGL","GSG","GLD"]
    asset_class_list = ["現金","日本株", "米国株", "先進国株", "新興国株", "米国債券", "先進国債券", "日本REIT", "米国REIT", "先進国REIT", "コモディティ", "金"]
    df_ticker = pd.DataFrame({"Ticker":ticker_list, "資産クラス":asset_class_list})
    df_ticker = df_ticker.set_index("Ticker")

    # 取得するデータ期間と間隔
    period = "11y" # 10年データが欲しいので長めにとって切り出す
    interval = "1d"

    # 為替データ(USD/JPY)の取得
    usdjpy, div = get_quote_dividend("USDJPY=X", period, interval)
    usdjpy = usdjpy.dropna()

    # 株価データの取得と保存
    quotes = pd.DataFrame(columns=df_ticker["資産クラス"])
    dividends = pd.DataFrame([])
    for ticker in df_ticker.index:
        # 現金はすべての値を1
        if ticker=="CASH":
            quote = bfill_10y(usdjpy).asfreq("1M") # 1か月毎データ
            quote[:]=1
            quotes[df_ticker.loc[ticker, "資産クラス"]]=quote
            continue

        # 株価と分配金を取得
        quote, div = get_quote_dividend(ticker, period, interval)

        # NYの場合、(1)14時間足して日本時間に合わせる、(2)価格を円換算
        if ".T" not in ticker:
            quote.index = quote.index + timedelta(hours=14)
            quote = conv_usd_to_jpy(quote, usdjpy)
            if div is not None:
                div.index = div.index + timedelta(hours=14)
                div = conv_usd_to_jpy(div, usdjpy)

        # 時刻情報を除去し日付のみにする
        quote.index = pd.to_datetime(quote.index.date)
        quote.index.name="timestamp"
        if div is not None:
            div.index = pd.to_datetime(div.index.date)
            div.index.name="timestamp"

        # 株価・分配金情報を格納
        quotes[df_ticker.loc[ticker, "資産クラス"]]=bfill_10y(quote).asfreq("1M")
        
        if div is not None:
            div.columns=[df_ticker.loc[ticker, "資産クラス"]]
            dividends = pd.merge(dividends, div, left_index=True, right_index=True, how="outer")
    # 分配金の月別リサンプル
    dividends = dividends.resample("M").sum()
    dividends = dividends.loc["2013-12-31":]

    # 分配金がない資産クラスは分配金0で列を作る
    for c in quotes.columns:
        if c not in dividends.columns:
            dividends[c] = np.zeros(len(dividends))
    
    return quotes, dividends

# 年率平均リターン・リスクの計算
def calc_score(df):
    # 年率平均リターン
    days=(df.index[-1] - df.index[0]).days
    n=int(days/365)
    annualized_return = ((df.iloc[-1] / df.iloc[0])**(1/n) -1)
    # 年率平均リスク
    monthly_return = df.pct_change(freq='1M') # 月次リターン
    monthly_risk = monthly_return.std() # 月次リターンの標準偏差
    annualized_risk = monthly_risk * np.sqrt(12) # 年率化

    score = pd.DataFrame([annualized_risk, annualized_return]).T
    score.columns = ["年率平均リスク", "年率平均リターン"]

    return score

# 毎月リバランスした際のリターン・リスクを計算
def calc_rebalanced_result(quotes, dividends, ratio):
    # ポートフォリオ比率が0の場合、悪い値を返す
    if np.sum(ratio)==0.0:
        score = calc_score(quotes["現金"])
        score[:] = [1.0, -1.0]
        return score

    # 初期設定
    amount = quotes.copy() # 資産毎の保有額
    amount[:] = 0
    num_of_stocks = amount.copy() # 資産毎の持株数
    portfolio = quotes.iloc[0].copy()
    portfolio[:] = ratio # ポートフォリオ比率
    portfolio /= portfolio.sum() # ポートフォリオの正規化

    # リバランス計算の実行
    start_date = quotes.index[0]
    for i in range(len(quotes.index)):
        d = quotes.index[i] # 日付

        # (1) 初月は定められた割合で購入。合計10000円とする
        if d==start_date:
            purchase = portfolio * 10000 # 1万円分買う
            num_of_stocks.loc[d] = purchase / quotes.loc[d] # 購入株数
            amount.loc[d] = num_of_stocks.loc[d] * quotes.loc[d] # 評価額
        else:
            # (2) 毎月得た分配金は再投資する。その後保有株数×価格でポートフォリオ割合を計算し、目標と乖離がある場合はリバランスする。
            # 分配金再投資
            reinvested = dividends.loc[d] * num_of_stocks.iloc[i-1] # 該当月の分配金で再投資する金額
            num_of_stocks.loc[d] = num_of_stocks.iloc[i-1] + reinvested / quotes.loc[d] # 金額分の株数を追加購入
            # 各資産クラスの割合を計算
            amount.loc[d] = num_of_stocks.loc[d] * quotes.loc[d]
            ratio = amount.loc[d] / amount.loc[d].sum()
            # ポートフォリオとの差
            diff_portfolio = portfolio - ratio
            # 購入額
            purchase = diff_portfolio * amount.loc[d].sum()
            num_of_stocks.loc[d] += purchase / quotes.loc[d]
            amount.loc[d] = num_of_stocks.loc[d] * quotes.loc[d]

    # (3) 最終日まで行ったら、トータル評価額からリターンとリスクを求める
    appraisal = amount.sum(axis=1)
    appraisal.name="資産評価額"
    score = calc_score(appraisal)

    # 評価データを保存
    result = pd.concat([portfolio, score.T]).T.values
    with open("result.csv", "a", encoding="utf-8-sig") as f:
        np.savetxt(f, result, delimiter=",", fmt="%0.4f")

    return score

# ポートフォリオ最適化問題の目的関数
class PortfolioOptimizationProblem(ElementwiseProblem):

    def __init__(self, quotes, dividends, **kwargs):
        super().__init__(n_var=12,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.zeros(12),
                         xu=np.ones(12),
                         **kwargs)
        self.quotes = quotes
        self.dividends = dividends

    def _evaluate(self, x, out, *args, **kwargs):
        score = calc_rebalanced_result(self.quotes, self.dividends, x)
        f1 = score.iloc[0,0] # リスクの最小化
        f2 = -1 * score.iloc[0,1] # 平均リターンの最大化
        
        out["F"] = [f1, f2]
        out["G"] = -1.0 * np.sum(x) + sys.float_info.epsilon # 変数の合計が0超過の制約

# ポートフォリオ最適化の実行
def run_portfolio_optimization():
    # 株価・分配金データの取得
    quotes, dividends = get_prices()

    # 最適化問題（マルチプロセスによる並列評価）
    n_proccess = 8
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)
    problem = PortfolioOptimizationProblem(quotes=quotes, 
                                           dividends=dividends, 
                                           elementwise_runner=runner)

    # NSGA-IIの設定
    algorithm = NSGA2(
        pop_size=50,
        sampling=np.diag(np.ones(12)), # 初期解を指定
        eliminate_duplicates=True
    )

    # 探索の実行
    termination = get_termination("n_gen", 100)
    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
    print(f"Threads: {n_proccess}, Execute time: {res.exec_time} sec")
    pool.close()

    # 結果の取得と保存
    X = res.X
    F = res.F
    pd.DataFrame(X).to_csv("X.csv", encoding="utf-8-sig", header=None, index=None)
    pd.DataFrame(F).to_csv("F.csv", encoding="utf-8-sig", header=None, index=None)

    # 解をソート
    F2 = F[np.argsort(F[:, 0])]
    F2[:,1] *= -1
    X2 = X[np.argsort(F[:, 0])]
    X2 = X2.T / X2.sum(axis=1)
    # シャープレシオの計算
    sharpratio = F2[:,1]/F2[:,0]

    # 最大リターン・最大シャープレシオの解
    num_max_sharpratio = np.argmax(sharpratio[2:]) # 最大シャープレシオの番号
    num_max_return = np.argmax(F2[:,1]) # 最大リターン
    num_balanced = np.where(F2[:,0]==F2[F2[:,0]>0.1,0].min())[0][0] # バランス
    num_low_risk = np.argmin(np.abs(F2[:,0]-0.05)) # 低リスク
    num_diversified=np.argmin(X2.var(axis=0))

    # パレート解以外の解の取得
    df = pd.read_csv("result.csv", encoding="utf-8-sig", header=None)
    X_all = df.iloc[:,:12].values
    F_all = df.iloc[:,12:].values

    # リスクvsリターン平面のプロット
    plt.figure(figsize=(7, 5))
    plt.scatter(F_all[:, 0]*100, F_all[:, 1]*100, s=3, facecolors="none", edgecolors="blue")
    plt.scatter(F2[:, 0]*100, F2[:, 1]*100, s=30, facecolors="red", edgecolors="red")
    #plt.scatter(F2[num_max_sharpratio,0]*100, F2[num_max_sharpratio,1]*-100, s=100, facecolors="none", edgecolors="green")
    #plt.scatter(F2[num_max_return,0]*100, F2[num_max_return,1]*-100, s=100, facecolors="none", edgecolors="green")
    plt.title("過去10年実績のポートフォリオ探索結果")
    plt.grid()
    plt.xlabel("リスク[%]")
    plt.ylabel("リターン[%]")
    plt.legend(["探索したポートフォリオ全体", "効率的フロンティア"])
    plt.xlim([-1,26])
    plt.ylim([-2.5,16])
    plt.show()

    # リスクvsシャープレシオのプロット
    plt.figure(figsize=(7, 1))
    plt.plot(F2[:,0]*100,sharpratio)
    plt.xlabel("リスク[%]")
    plt.ylabel("シャープレシオ")
    plt.xlim([-1,26])
    plt.grid()
    plt.show()

    # 探索推移
    ref_point = np.array([np.max(F_all[:,0]), np.min(F_all[:,1])]) # 参照点はすべての解のリスク最大リターン最小の点
    ind = HV(ref_point=ref_point)
    hv = [ind(np.array([p.F for p in h.pop])) for h in res.history]
    plt.plot(hv)
    plt.xlabel("世代数")
    plt.ylabel("探索の進行度(Hypervolume)")
    plt.xlim([-5, 100])
    plt.grid()

    # 最大リターン/バランス/低リスクポートフォリオの表示
    portfolio = pd.DataFrame(X2[:,[num_max_return, num_max_sharpratio, num_balanced, num_diversified, num_low_risk]])
    portfolio.index = quotes.columns
    portfolio = portfolio.round(3)*100
    portfolio.columns = ["最大リターン", "最大シャープレシオ", "バランス", "分散", "低リスク"]
    score = pd.DataFrame(F2[[num_max_return, num_max_sharpratio, num_balanced, num_diversified, num_low_risk]]*100)
    score.index=portfolio.columns
    score.columns = ["リスク", "平均リターン"]
    score = score
    print("効率フロンティア上のポートフォリオと成績の例")
    print(portfolio.round(3))
    print(score.round(1))
    
if __name__ == "__main__":
    run_portfolio_optimization()
