from src.database.database import Base, SessionLocal, engine
from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String, func, desc

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    session = SessionLocal()
    ans = []
    for user in (
        session.query(User.country, User.os, func.count().label("count"))
        .filter(User.exp_group == 3)
        .group_by(User.country, User.os)
        .having(func.count() > 100)
        .order_by(desc("count"))
        .all()
    ):
        ans.append((user.country, user.os, user.count))
    print(ans)
