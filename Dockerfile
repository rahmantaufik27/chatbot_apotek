FROM python:3.10 as base
FROM base as builder
RUN mkdir /install
WORKDIR /install
COPY ChatterBot-1.1.0.tar.gz ChatterBot-1.1.0.tar.gz
COPY requirements.txt .
COPY prod_requirements.txt .
RUN apt update && pip install --upgrade pip && pip install --prefix="/install" -r requirements.txt
FROM base
COPY --from=builder /install /usr/local
COPY . /app
WORKDIR /app
ENTRYPOINT [ "sh", "entrypoint.sh" ]