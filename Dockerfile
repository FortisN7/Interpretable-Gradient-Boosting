FROM python
COPY . /app
CMD python -u /app/app.py
# Add pip eventually so I can install plugins like pandas and numpy