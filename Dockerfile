FROM continuumio/miniconda
RUN conda create -n "moseq2" python=3.6 -y
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "source activate moseq2" > ~/.bashrc
ENV PATH /opt/conda/envs/moseq2/bin:$PATH
ENV SRC /src
ENV PYTHONPATH /src
RUN mkdir -p $SRC

COPY . $SRC/moseq2-extract

RUN pip install -e $SRC/moseq2-extract