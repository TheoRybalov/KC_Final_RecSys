from src.database.database import Base, SessionLocal, engine
from src.database.tables.table_user import User
from src.database.tables.table_post import Post
from sqlalchemy.orm import relationship
from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String, func, desc

class Feed(Base):
    __tablename__ = "feed_action"
    user_id = Column(
        Integer, ForeignKey(User.id), primary_key=True, name="user_id"
        )
    user = relationship("User")
    post_id = Column(
        Integer, ForeignKey(Post.id), primary_key=True, name="post_id"
        )
    post = relationship("Post")
    action = Column(String)
    time = Column(TIMESTAMP)