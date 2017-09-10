import random

from sqlalchemy import Column, String, Integer
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

DB_CONFIG = {
    'drivername': 'postgresql',
    'username': '',
    'password': '',
    'host': 'localhost',
    'port': 5432,
    'database': 'ngram'
}

engine = create_engine(URL(**DB_CONFIG))
Session = sessionmaker()
Session.configure(bind=engine)


class Data(Base):
    __tablename__ = "ngrams"
    id = Column(Integer, primary_key=True)
    key = Column(String, index=True)
    value = Column(String)
    cnt = Column(Integer)
    ngram_type = Column(Integer, index=True)


class NgramModel(object):
    def __init__(self, n=3):
        self.n = n

        self.s = Session()

    def forward(self, inputs: tuple):
        if len(inputs) is not self.n - 1:
            return False

        r = self.s.query(Data.value, Data.cnt).filter(Data.key == str(inputs), Data.ngram_type == self.n).order_by(Data.cnt.desc()).all()
        if r:
            res = []
            max_v = r[0][1]

            ratio = round(max_v / sum([_[1] for _ in r]) * 100, 2)

            for _ in r:
                if _[1] == max_v:
                    res.append(_[0])

            val = random.choice(res)

            return (val, ratio)

        else:
            return False

    def __del__(self):
        self.s.close()


if __name__ == '__main__':
    m = NgramModel()
    r = m.forward(("'", 'use'))
    print(r)


