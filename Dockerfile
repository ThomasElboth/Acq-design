FROM python:3.7-slim

#not sure how many of these are needed....
RUN apt-get update && apt-get install -y \
  dos2unix \
  apt-utils \
  gcc

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 80
RUN mkdir ~/.streamlit
RUN mkdir ~/Gun_Database
RUN mkdir ~/Vib_Database
RUN cp dither_explained.pptx ~/dither_explained.pptx
RUN cp acq_plots.py ~/acq_plots.py
RUN cp vib_directivity.py ~/vib_directivity.py
RUN cp make_dithers.py ~/make_dithers.py
RUN cp make_sweeps.py ~/make_sweeps.py
RUN cp Gun_Database/* ~/Gun_Database
RUN cp Vib_Database/* ~/Vib_Database
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["acqplotterAPP.py"]
