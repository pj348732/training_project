import os
import pandas as pd
import datetime
import numpy as np
from datetime import timedelta
import os
from dateutil.relativedelta import relativedelta


def get_slurm_env(name):
    value = os.getenv(name)
    if value is None:
        if name == 'SLURM_ARRAY_TASK_ID' or name == 'SLURM_PROCID':
            return 0
        else:
            return 1
    else:
        return value


def get_all_stocks_by_day(day_i):
    mta_df = pd.read_parquet('/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=day_i))
    return list(sorted(list(mta_df.skey.unique())))


def iter_time_range(start_day, end_day):
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = datetime.datetime.strptime(str(end_day), "%Y%m%d")
    delta = edate - sdate
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        yield int(day.strftime("%Y%m%d"))


def get_trade_days():
    trade_days = list(sorted(
        [int(d) for d in os.listdir('/b/sta_feat_eq_cn/sta_feat_1_2_l2/{group}/'.format(group='LabelTm'))
         if d != 'Label']))
    return trade_days


def get_weekday(day):
    sdate = datetime.datetime.strptime(str(day), "%Y%m%d")
    return int(sdate.weekday())


def get_session_id(minute):
    if 0 <= minute <= 15:
        return 0
    elif 15 < minute <= 30:
        return 1
    elif 30 < minute <= 220:
        return 2
    else:
        return 3


def get_month_day(x):
    sdate = datetime.datetime.strptime(str(x), "%Y%m%d")
    return int(sdate.day)


def time_to_minute(t):
    hr = int(t / 1e4)
    minute = int((t - hr * 10000) / 100)
    mSinceOpen = (hr - 9) * 60 + (minute - 30)
    if (t >= 93000) and (t < 113000):
        return mSinceOpen
    elif (t >= 130000) and (t < 145700):
        return mSinceOpen - 90
    else:
        return -1


def get_encode_path(day_i):
    e_path = f'/b/home/pengfei_ji/mnt_files/{day_i}/'
    try:
        if not os.path.exists(e_path):
            os.mkdir(e_path)
    except:
        return e_path
    return e_path


def after_one_month(start_day):
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = sdate + relativedelta(months=1)
    return int(edate.strftime("%Y%m%d"))


def iter_sec_range(start_time, end_time):
    s_time = datetime.datetime.strptime(start_time, '%H:%M:%S')
    e_time = datetime.datetime.strptime(end_time, '%H:%M:%S')
    delta = e_time - s_time
    for i in range(delta.seconds + 1):
        clock_time = s_time + timedelta(seconds=i)
        yield int(clock_time.strftime("%H%M%S"))


def n_day_before(start_day, nday):
    sdate = datetime.datetime.strptime(str(start_day), "%Y%m%d")
    edate = sdate - timedelta(days=nday)
    return int(edate.strftime("%Y%m%d"))


def to_str_date(x):
    x = int(x)
    return (datetime.date(1899, 12, 30) + datetime.timedelta(days=x)).strftime("%Y%m%d")


def get_r2_score(preds, gts):
    gt_mean = np.mean(gts)
    SSres = sum(map(lambda x: (x[0] - x[1]) ** 2, zip(gts, preds)))
    SStot = sum([(x - gt_mean) ** 2 for x in gts])
    return 1 - (SSres / SStot)


def to_int_date(x):
    x = str(x)
    if '-' in x:
        dtInfo = x.split('-')
        return (datetime.date(int(dtInfo[0]), int(dtInfo[1]), int(dtInfo[2])) - datetime.date(1899, 12, 30)).days
    else:
        if len(x) == 8:
            dtInfo = x
            return (datetime.date(int(dtInfo[:4]), int(dtInfo[4:6]), int(dtInfo[6:8])) - datetime.date(1899, 12,
                                                                                                       30)).days
        else:
            return x
