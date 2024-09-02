FROM python:3.10
ADD ./run_model.py .
ADD ./data/r2_gse62564_GSVA_Metadata_selected.csv .
ADD ./requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "run_model.py"]

# to build the image and deploy use:
    # docker build -t python-model .
    # docker run -it python-model