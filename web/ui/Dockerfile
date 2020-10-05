### BUILD ###

FROM node:alpine AS build
ARG IDLP_HOST=localhost
RUN echo ${IDLP_HOST}
ENV IDLP_HOST=${IDLP_HOST}
RUN echo $IDLP_HOST
RUN mkdir /code
WORKDIR /code
COPY . .
RUN npm install
RUN npm run build -- --prod --output-hashing all

### RUN ###

FROM nginx:stable
COPY ./ui.nginx.conf /etc/nginx/conf.d/ui.nginx.conf
RUN mkdir /code
WORKDIR /code

COPY --from=build /code/dist/ui /usr/share/nginx/html
