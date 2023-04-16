FROM python:3.10
RUN pip install pandas numpy xgboost shap matplotlib
COPY . /app
CMD python -u /app/app.py