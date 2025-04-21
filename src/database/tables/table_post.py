from database.database import Base, SessionLocal, engine
from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String

class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    session = SessionLocal()
    ans = []
    for post in (
        session.query(Post)
        .filter(Post.topic == "business")
        .order_by(Post.id.desc())
        .limit(10)
        .all()
    ):
        ans.append(post.id)
    print(ans)