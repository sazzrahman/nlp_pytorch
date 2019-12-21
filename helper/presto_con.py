import pandas as pd
from pyhive import presto

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from sqlalchemy.sql import text

class PrestoQuery(object):
    def __init__(self):
        self.cursor = presto.connect(host="cw-presto-coordinator-east.lkqd.com", port=8080).cursor()
        self.description = None
        self.raw = None
        self.name = None
        self.dtype = None

    def pquery(self, query):
        self.cursor.execute(query)
        self.description = self.cursor.description
        self.parse_description()
        self.raw = self.cursor.fetchall()
        df = pd.DataFrame(self.raw,columns=self.name)
        return df


    def parse_description(self):
        self.name, self.dtype, _, _, _, _, _ = zip(*self.description)




class PrestoAlchemy(object):
    def __init__(self):
        self.engine = create_engine('presto://cw-presto-coordinator-east.lkqd.com:8080/hive/default')
        self.description = None

    def pquery(self,query):
        with self.engine.connect() as con:
            rs = con.execute(query)
            return pd.DataFrame(rs)
