"""Validate expected Part A checkpoints."""
import pandas as pd
import sqlite3
from math import isclose
c=pd.read_csv('data/clean/customers.csv'); o=pd.read_csv('data/clean/orders.csv')
assert len(c)==1200 and c.city.nunique()==6
assert c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}
assert c.age.isna().sum()==0 and c.age.min()==18 and c.age.max()==60
assert len(o)==12309 and o.order_id.nunique()==12309
assert o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199}
assert o.order_amount.isna().sum()==0
assert o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0
assert o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199
assert isclose(o.loc[o.status=='completed','order_amount'].sum(),2998668.80,abs_tol=.01)
with sqlite3.connect('data/instamart.db') as con:
    tables={row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'customers','products','orders'} <= tables
print('ALL CHECKPOINTS PASSED')
